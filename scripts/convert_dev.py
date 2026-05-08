"""
Convert dev.csv (raw Dianping reviews with aspect labels) to the project's expected format.

Usage: python scripts/convert_dev.py

Produces data/reviews.csv with columns:
    review_id, user_id, dish_id, dish_name, restaurant, review_text, rating, date
"""

import csv
import os
import re
import sys

import pandas as pd

# Ensure project root is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

SOURCE_PATH = os.path.join(project_root, "data", "dev.csv")
TARGET_PATH = os.path.join(project_root, "data", "reviews.csv")
NUM_ROWS = None  # Process all rows


def extract_restaurant_name(text: str) -> str:
    """Extract restaurant name from the beginning of a review text."""
    if not isinstance(text, str) or not text.strip():
        return "未知餐厅"

    # Pattern 1: 【Restaurant Name】
    m = re.search(r"【(.+?)】", text)
    if m:
        return m.group(1).strip()

    # Pattern 2: First sentence often contains the restaurant name
    # Look for common patterns like "XXX店", "XXX餐厅", etc.
    first_part = text[:200]
    m = re.search(
        r"((?:[^\s，。,\.!！;；、]){2,10}?"
        r"(?:店|餐厅|饭馆|酒楼|火锅|料理|食府|大排档|小吃|烧烤|"
        r"面馆|粉店|粥店|茶楼|咖啡|面包|甜品|自助|铁板烧|"
        r"麻辣烫|麻辣香锅|鸭血粉丝|煲仔饭|烤鸭|鸡排|酸菜鱼))",
        first_part,
    )
    if m:
        name = m.group(1).strip()
        # Remove leading noise words
        name = re.sub(r"^[来到去在了是的有这和那我说他她它]*", "", name)
        if len(name) >= 2:
            return name

    # Pattern 3: "来到XXX", "去了XXX", "在XXX吃"
    m = re.search(r"(?:来到|去了|去啦|在|试试)((?:[^\s，。,\.!！;；、]){2,10}?)(?:吃|饭|店|用餐|尝试|品尝|打卡|觅食|探店)", first_part)
    if m:
        return m.group(1).strip()

    return "未知餐厅"


def run():
    print(f"Reading {SOURCE_PATH}...")
    df = pd.read_csv(SOURCE_PATH, nrows=NUM_ROWS)
    print(f"Loaded {len(df)} rows")

    # Try to import project's dish extraction
    extract_dish_name = None
    try:
        from scripts.convert_dianping import extract_dish_name as _extract
        extract_dish_name = _extract
        print("Using project dish name extractor")
    except ImportError:
        print("Warning: Could not import dish name extractor, will use fallback")

    output_rows = []
    dish_counter = {}

    for idx, (_, row) in enumerate(df.iterrows()):
        review_text = str(row["review"]).strip() if pd.notna(row.get("review")) else ""
        if len(review_text) < 10:
            continue

        rating_raw = row.get("star")
        if pd.isna(rating_raw):
            continue
        rating = int(round(float(rating_raw)))

        restaurant = extract_restaurant_name(review_text)

        # Extract dish name
        dish_name = None
        if extract_dish_name:
            dish_name = extract_dish_name(review_text, restaurant)
        if not dish_name:
            dish_name = restaurant  # fallback to restaurant name

        # Collect unique dishes
        if dish_name not in dish_counter:
            dish_counter[dish_name] = len(dish_counter) + 1

        output_rows.append({
            "review_id": len(output_rows) + 1,
            "user_id": f"U{int(row['id'])}",
            "dish_id": f"D{dish_counter[dish_name]:05d}",
            "dish_name": dish_name,
            "restaurant": restaurant,
            "review_text": review_text,
            "rating": rating,
            "date": "2024-01-01",  # No date in source data, use placeholder
        })

    output_df = pd.DataFrame(output_rows, columns=[
        "review_id", "user_id", "dish_id", "dish_name",
        "restaurant", "review_text", "rating", "date",
    ])

    os.makedirs(os.path.dirname(TARGET_PATH), exist_ok=True)
    output_df.to_csv(TARGET_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(output_df)} rows to {TARGET_PATH}")
    print(f"  Unique users: {output_df['user_id'].nunique()}")
    print(f"  Unique dishes: {output_df['dish_id'].nunique()}")
    print(f"  Unique restaurants: {output_df['restaurant'].nunique()}")
    print(f"  Rating distribution: {dict(output_df['rating'].value_counts().sort_index())}")


if __name__ == "__main__":
    run()
