"""
Convert the Dianping restaurant review dataset to the format expected by this project.

Usage: python scripts/convert_dianping.py [--target-rows N] [--seed SEED]

Produces data/reviews.csv with columns:
    review_id, user_id, dish_id, dish_name, restaurant, review_text, rating, date
"""

import argparse
import csv
import logging
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_DIR = r"E:\BaiduNetdiskDownload\文本挖掘数据集\yf_dianping\yf_dianping"
TARGET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reviews.csv")
DEFAULT_TARGET_ROWS = 75_000
CHUNK_SIZE = 100_000
MIN_COMMENT_LEN = 10

# ---------------------------------------------------------------------------
# Food context words for filtering non-dining reviews
# ---------------------------------------------------------------------------
# The Dianping dataset includes reviews for ALL business types (museums,
# supermarkets, electronics stores, etc.), not just restaurants.  Reviews
# must contain enough food/dining signals to pass this filter.
FOOD_SIGNALS = {
    # Eating & drinking
    "吃", "喝", "食", "饮", "尝", "啃", "吞",
    # Taste & texture
    "味道", "口味", "口感", "好吃", "美味", "好味", "好食", "好喝",
    "香", "辣", "甜", "酸", "鲜", "嫩", "酥", "脆", "咸", "淡", "麻",
    "入味", "够味", "对味", "正宗", "地道",
    # Food ingredients
    "菜", "汤", "肉", "鱼", "虾", "蟹", "鸡", "鸭", "牛", "猪", "羊", "鹅",
    "蛋", "豆腐", "豆花", "饭", "粥", "饼", "饺", "面", "粉",
    # Cooking methods
    "烤", "炒", "蒸", "煮", "炸", "煎", "炖", "煲", "烧", "涮", "烫",
    "卤", "熏", "焖", "焗", "烩",
    # Dining context
    "火锅", "自助", "早茶", "下午茶", "宵夜", "夜宵",
    "招牌菜", "特色菜", "推荐菜", "拿手菜", "菜品", "出品",
    "菜单", "点菜", "上菜", "点单", "等位", "排号",
    # Restaurant types
    "餐厅", "饭馆", "酒楼", "大排档", "食堂", "食府", "小吃店",
    # Specific cuisines / dishes
    "烧烤", "刺身", "寿司", "沙拉", "披萨", "汉堡",
    "川菜", "粤菜", "湘菜", "鲁菜", "苏菜", "浙菜", "闽菜", "徽菜",
    "本帮菜", "东北菜", "西北菜",
}
NON_FOOD_SIGNALS = {
    # Shopping / retail (strong signals)
    "超市", "购物", "百货", "商场", "广场", "大卖场",
    # Electronics
    "电器", "手机", "电脑", "数码", "家电", "洗衣机", "冰箱",
    # Tourism
    "旅游", "景点", "博物馆", "门票", "导游", "旅行团", "参观",
    # Other retail
    "家具", "家私", "服装", "衣服", "鞋子", "眼镜", "乐器",
    "钢琴", "吉他", "配眼镜",
}

# Words that match food-suffix patterns but are NOT dish names.
# These pass the food-suffix check (e.g. end with 面/粉/排/块/团/包/菜/饭)
# but refer to non-food concepts.
NON_FOOD_BLACKLIST: set[str] = {
    # --- 面 = face/side/surface, not noodle ---
    "里面", "外面", "方面", "对面", "前面", "后面", "下面", "上面",
    "地面", "表面", "门面", "店面", "铺面", "点面", "漆面", "碑面",
    "场面", "体面", "全面", "负面", "方方面面", "入面", "出面",
    "斜对面", "装修门面",
    # --- 粉 = powder, not noodle ---
    "洗衣粉",
    # --- 排 = row/queue, not rib ---
    "排队", "有排", "近排", "前排", "要排", "队排",
    # --- 块 = yuan (money), not piece ---
    "一块", "五块", "十多块", "二十块", "几十块", "几百块", "千块",
    # --- 团 = group, not dumpling ---
    "旅游团", "旅行团", "带团", "乐团", "集团",
    # --- 包 = bag/package, not bun ---
    "包包", "荷包",
    # --- 煲 = cooker/appliance, not claypot dish ---
    "电饭煲", "压力锅",
    # --- Non-food products ---
    "图片", "照片", "拍照片", "螺丝", "镜片", "平底锅",
    # --- Actions, not dishes ---
    "买菜", "煮饭", "吃饭", "吃完饭", "食饭", "喝茶", "吃火锅",
    "上菜", "点菜", "订菜", "喝汤", "吃面", "吃粉", "吃鸡",
    # --- Meal occasions / categories (too broad, not specific dishes) ---
    "晚饭", "早餐", "午饭", "中饭", "盒饭", "饭菜",
    # --- Non-food misc ---
    "知识面", "街面", "花样年华", "不饭",
}

# Timestamp bounds (plausible Dianping era, 2000—2026)
TS_MIN = 946684800000   # 2000-01-01 in ms
TS_MAX = 1798755200000  # 2027-01-01 in ms

FOOD_SUFFIXES = {
    # Meats & proteins
    "肉", "鱼", "虾", "蟹", "鸡", "鸭", "鹅", "牛", "羊", "猪",
    # Soups, noodles, staples
    "汤", "面", "饭", "粉", "粥", "饼", "包", "饺", "糕", "团",
    # Dishes, pots, skewers
    "菜", "锅", "煲", "串", "翅", "爪", "排", "骨", "柳", "片",
    "扒", "丁", "丝", "块", "丸", "卷", "酥", "羹", "汁", "酱",
    # Bean & egg & veg
    "豆腐", "豆花", "蛋", "菌", "菇", "笋", "藕", "茄", "瓜",
    # Combo terms (longer, check first)
    "虾仁", "鱼片", "牛肉", "鸡肉", "猪肉", "羊肉",
    # Drinks
    "奶茶", "咖啡", "果汁", "啤酒", "红酒", "白酒", "茶",
    # Desserts & baked
    "蛋糕", "面包", "甜品", "冰淇淋", "巧克力", "糖果",
    # Foreign dishes
    "沙拉", "刺身", "寿司", "披萨", "汉堡", "薯条",
}

VALID_POS = {"n", "nr", "nz", "ng", "vn"}
MIN_DISH_LEN = 2
MAX_DISH_LEN = 10

# Characters commonly found in food terms (used for food-char ratio scoring)
ALL_FOOD_CHARS: set[str] = set()
for _s in FOOD_SUFFIXES:
    ALL_FOOD_CHARS.update(_s)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_restaurant_names(source_dir: str) -> dict[int, str]:
    """Build {restId: restaurant_name} dict, with fallback for empty names."""
    rest_path = os.path.join(source_dir, "restaurants.csv")
    rest_df = pd.read_csv(rest_path)
    names = {}
    for _, row in rest_df.iterrows():
        rid = int(row["restId"])
        name = str(row["name"]).strip() if pd.notna(row["name"]) else ""
        names[rid] = name if name else f"餐厅_{rid}"
    log.info("Loaded %d restaurant names (%d with empty names)",
             len(names), sum(1 for v in names.values() if v.startswith("餐厅_")))
    return names


def derive_rating(row: pd.Series) -> int | None:
    """
    Derive a 1-5 integer rating.
    Uses explicit rating if present; otherwise averages the three sub-ratings.
    Returns None when no rating signal is available.
    """
    if pd.notna(row.get("rating")) and 1 <= row["rating"] <= 5:
        return int(row["rating"])
    subs = []
    for col in ("rating_flavor", "rating_env", "rating_service"):
        val = row.get(col)
        if pd.notna(val) and 1 <= val <= 5:
            subs.append(val)
    if subs:
        return int(round(sum(subs) / len(subs)))
    return None


def convert_timestamp(ts: float) -> str | None:
    """Convert Unix ms timestamp to YYYY-MM-DD string.  Returns None on out-of-range."""
    try:
        if pd.isna(ts) or ts <= 0:
            return None
        if ts < TS_MIN or ts > TS_MAX:
            return None
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return None


def _is_food_review(comment: str) -> bool:
    """True if the review is about food/dining (not supermarket, museum, etc.)."""
    food_cnt = sum(1 for w in FOOD_SIGNALS if w in comment)
    non_food_cnt = sum(1 for w in NON_FOOD_SIGNALS if w in comment)
    return food_cnt >= 2 and food_cnt > non_food_cnt


def _ends_with_food_suffix(token: str) -> bool:
    """True if token ends with any food suffix and is at least MIN_DISH_LEN chars."""
    if len(token) < MIN_DISH_LEN:
        return False
    for suffix in sorted(FOOD_SUFFIXES, key=len, reverse=True):
        if token.endswith(suffix):
            return True
    return False


def _extract_candidates_pos(comment: str) -> list[str]:
    """Layer 1: Jieba POS tagging + food-suffix matching, with compound merging."""
    import jieba.posseg as pseg

    tagged = list(pseg.cut(comment))
    if not tagged:
        return []

    # Collect all food-related tokens with their positions
    food_items: list[tuple[int, str]] = []  # (index, token)
    for idx, (word, flag) in enumerate(tagged):
        if flag not in VALID_POS:
            continue
        if not _ends_with_food_suffix(word):
            continue
        if MIN_DISH_LEN <= len(word) <= MAX_DISH_LEN:
            food_items.append((idx, word))

    if not food_items:
        return []

    # Merge consecutive food tokens into compound names
    # e.g. (0, "红烧") + (1, "牛肉") -> check if "红烧牛肉" exists in comment
    merged: list[str] = []
    i = 0
    while i < len(food_items):
        group = [food_items[i][1]]
        indices = [food_items[i][0]]
        # Extend group while tokens are consecutive in the original tagged sequence
        for j in range(i + 1, len(food_items)):
            if food_items[j][0] == indices[-1] + 1:
                group.append(food_items[j][1])
                indices.append(food_items[j][0])
            else:
                break
        # Create candidate from each prefix of the group
        for k in range(len(group)):
            combined = "".join(group[:k + 1])
            if combined in comment and MIN_DISH_LEN <= len(combined) <= MAX_DISH_LEN:
                merged.append(combined)
        i += len(group)

    # Flatten: also include individual tokens as candidates
    result = list({w for _, w in food_items} | set(merged))
    return result[:15]


def _extract_candidates_keywords(comment: str) -> list[str]:
    """Layer 2: Use the project's TF-IDF keyword extraction + food filtering."""
    try:
        from app.nlp.keyword import extract_keywords
        kw = extract_keywords(comment, topk=15, method="tfidf")
    except Exception:
        return []
    candidates = []
    for word, _weight in kw:
        if _ends_with_food_suffix(word) and MIN_DISH_LEN <= len(word) <= MAX_DISH_LEN:
            candidates.append(word)
    return candidates


def _score_candidate(candidate: str, comment: str, kw_dict: dict[str, float]) -> float:
    """Score a dish name candidate. Higher = better."""
    score = 0.0

    # Position: earlier mentions score higher
    pos = comment.find(candidate)
    if pos >= 0:
        score += max(0.0, 1.0 - pos / max(1, len(comment)))

    # Keyword weight bonus
    if candidate in kw_dict:
        score += kw_dict[candidate] * 2.0

    # Length bonus: real dish names are usually 2-6 chars
    if 2 <= len(candidate) <= 6:
        score += 0.5
    elif len(candidate) <= MAX_DISH_LEN:
        score += 0.2

    # Food character ratio
    food_ratio = sum(1 for ch in candidate if ch in ALL_FOOD_CHARS) / len(candidate)
    score += food_ratio * 0.5

    return score


def _fallback_dish(restaurant_name: str) -> str | None:
    """Return restaurant_name as dish proxy, or None if it looks like a branch ID."""
    if not restaurant_name:
        return None
    # Skip generic placeholders
    if restaurant_name.startswith("餐厅_"):
        return None
    # Skip names with branch/location info in brackets
    if "(" in restaurant_name or "（" in restaurant_name:
        return None
    # Skip blacklisted words
    if restaurant_name in NON_FOOD_BLACKLIST:
        return None
    return restaurant_name


def extract_dish_name(comment: str, restaurant_name: str) -> str | None:
    """
    Extract a single best-guess dish name from a Chinese restaurant review.
    Returns None only when no fallback is available.
    """
    if not comment or not isinstance(comment, str):
        return restaurant_name if restaurant_name else None

    # Build keyword weight dictionary
    kw_dict: dict[str, float] = {}
    try:
        from app.nlp.keyword import extract_keywords
        for word, weight in extract_keywords(comment, topk=15, method="tfidf"):
            kw_dict[word] = weight
    except Exception:
        pass

    # Collect candidates from both layers
    all_candidates: list[str] = []
    seen: set[str] = set()

    for cand in _extract_candidates_pos(comment):
        if cand not in seen:
            seen.add(cand)
            all_candidates.append(cand)

    for cand in _extract_candidates_keywords(comment):
        if cand not in seen:
            seen.add(cand)
            all_candidates.append(cand)

    # Remove blacklisted non-food words
    all_candidates = [c for c in all_candidates if c not in NON_FOOD_BLACKLIST]

    # Remove candidates that look like restaurant names with branch info
    all_candidates = [
        c for c in all_candidates
        if "(" not in c and "（" not in c and not c.startswith("餐厅_")
    ]

    if not all_candidates:
        return _fallback_dish(restaurant_name)

    # Score and pick best
    best = max(all_candidates, key=lambda c: _score_candidate(c, comment, kw_dict))
    best_score = _score_candidate(best, comment, kw_dict)

    # Confidence threshold
    if best_score >= 0.3:
        return best

    return _fallback_dish(restaurant_name)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(source_dir: str, target_path: str, target_rows: int, seed: int) -> None:
    random.seed(seed)

    # --- Stage 1: Restaurant name lookup ---
    rest_name_map = load_restaurant_names(source_dir)

    # --- Stage 2 & 3: Chunked read + filter + reservoir sampling ---
    # Prefer the extracted CSV; fall back to the zip
    ratings_csv = os.path.join(source_dir, "ratings", "ratings.csv")
    ratings_zip = os.path.join(source_dir, "ratings.zip")
    if os.path.exists(ratings_csv):
        ratings_path = ratings_csv
    elif os.path.exists(ratings_zip):
        ratings_path = ratings_zip
    else:
        log.error("Neither ratings/ratings.csv nor ratings.zip found in %s", source_dir)
        sys.exit(1)
    log.info("Using ratings data from: %s", ratings_path)

    reservoir: list[tuple] = []
    valid_count = 0  # number of valid rows encountered (passed filters)

    log.info("Reading ratings.zip in %d-row chunks (target: %d rows)...", CHUNK_SIZE, target_rows)

    reader = pd.read_csv(
        ratings_path,
        chunksize=CHUNK_SIZE,
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="skip",
    )

    for chunk_no, chunk in enumerate(reader):
        for _, row in chunk.iterrows():
            # Filter: comment must exist and be long enough
            comment = str(row["comment"]).strip() if pd.notna(row.get("comment")) else ""
            if len(comment) < MIN_COMMENT_LEN:
                continue

            # Filter: must be a food/dining review (not supermarket, museum, etc.)
            if not _is_food_review(comment):
                continue

            # Filter: must have a usable rating
            rating = derive_rating(row)
            if rating is None:
                continue

            # Filter: timestamp must convert
            date = convert_timestamp(row["timestamp"])
            if date is None:
                continue

            valid_count += 1

            # Standard reservoir sampling (uniform probability)
            if len(reservoir) < target_rows:
                reservoir.append((row, comment, rating, date))
            else:
                j = random.randint(0, valid_count - 1)
                if j < target_rows:
                    reservoir[j] = (row, comment, rating, date)

        if (chunk_no + 1) % 10 == 0:
            log.info("  Chunk %d: %d valid rows, reservoir size=%d",
                     chunk_no + 1, valid_count, len(reservoir))

    log.info("Sampling complete: %d valid rows found, %d in reservoir",
             valid_count, len(reservoir))

    # --- Stage 4: Dish name extraction ---
    log.info("Extracting dish names for %d sampled reviews...", len(reservoir))
    dish_counter: Counter = Counter()
    extracted_dishes: list[str | None] = []
    fallback_to_restaurant = 0
    skipped = 0

    for idx, (row, comment, rating, date) in enumerate(reservoir):
        rest_id = int(row["restId"])
        restaurant = rest_name_map.get(rest_id, f"餐厅_{rest_id}")

        dish_name = extract_dish_name(comment, restaurant)
        if dish_name is None:
            skipped += 1
            extracted_dishes.append(None)
            continue

        if dish_name == restaurant:
            fallback_to_restaurant += 1

        extracted_dishes.append(dish_name)
        dish_counter[dish_name] += 1

        if (idx + 1) % 10_000 == 0:
            log.info("  Extracted dish names for %d/%d reviews...", idx + 1, len(reservoir))

    log.info("Dish extraction: %d extracted, %d fallback-to-restaurant, %d skipped, %d unique dishes",
             len(extracted_dishes) - skipped, fallback_to_restaurant, skipped, len(dish_counter))

    # --- Stage 5: Assign dish IDs ---
    log.info("Assigning dish IDs for %d unique dish names...", len(dish_counter))
    unique_dishes = sorted(dish_counter.keys())
    dish_id_map = {name: f"D{idx + 1:05d}" for idx, name in enumerate(unique_dishes)}
    log.info("Dish ID range: %s — %s",
             dish_id_map[unique_dishes[0]] if unique_dishes else "N/A",
             dish_id_map[unique_dishes[-1]] if unique_dishes else "N/A")

    # --- Stage 6: Assemble and write output ---
    log.info("Assembling output DataFrame...")
    output_rows = []
    for idx, row_comment_rating_date in enumerate(reservoir):
        row, comment, rating, date = row_comment_rating_date
        dish_name = extracted_dishes[idx]
        if dish_name is None:
            continue

        rest_id = int(row["restId"])
        restaurant = rest_name_map.get(rest_id, f"餐厅_{rest_id}")

        output_rows.append({
            "review_id": len(output_rows) + 1,
            "user_id": f"U{int(row['userId'])}",
            "dish_id": dish_id_map[dish_name],
            "dish_name": dish_name,
            "restaurant": restaurant,
            "review_text": comment,
            "rating": rating,
            "date": date,
        })

    output_df = pd.DataFrame(output_rows, columns=[
        "review_id", "user_id", "dish_id", "dish_name",
        "restaurant", "review_text", "rating", "date",
    ])

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    output_df.to_csv(target_path, index=False, encoding="utf-8")
    log.info("Wrote %d rows to %s", len(output_df), target_path)

    # --- Summary ---
    log.info("========== Summary ==========")
    log.info("Rows written:            %d", len(output_df))
    log.info("Unique users:            %d", output_df["user_id"].nunique())
    log.info("Unique dishes:           %d", output_df["dish_id"].nunique())
    log.info("Unique restaurants:      %d", output_df["restaurant"].nunique())
    log.info("Rating distribution:     %s", dict(output_df["rating"].value_counts().sort_index()))
    log.info("Date range:              %s — %s", output_df["date"].min(), output_df["date"].max())
    log.info("Average comment length:  %.0f chars", output_df["review_text"].str.len().mean())
    log.info("==============================")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Dianping dataset to project format")
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS,
                        help=f"Target number of output rows (default: {DEFAULT_TARGET_ROWS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--source-dir", type=str, default=SOURCE_DIR,
                        help="Path to the Dianping dataset directory")
    parser.add_argument("--target-path", type=str, default=TARGET_PATH,
                        help="Output CSV path")
    args = parser.parse_args()

    # Ensure project root is on sys.path so app imports work
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    run(args.source_dir, args.target_path, args.target_rows, args.seed)
