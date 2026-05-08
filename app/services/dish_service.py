from collections import defaultdict
import pandas as pd
from config import SENTIMENT_FLOOR


def build_dish_profiles(reviews_df):
    """
    Aggregate reviews by dish_id to build dish taste profiles.
    Returns DataFrame with columns: dish_id, dish_name, restaurant, tag_text, tags, review_count
    """
    dishes = []

    for dish_id, group in reviews_df.groupby('dish_id'):
        keyword_stats = defaultdict(lambda: {'freq': 0, 'sentiments': []})

        for _, row in group.iterrows():
            for kw in row['keywords']:
                word = kw['word']
                keyword_stats[word]['freq'] += 1
                keyword_stats[word]['sentiments'].append(kw['sentiment'])

        # Build tag list
        tags = []
        tag_text_parts = []
        for word, stats in keyword_stats.items():
            avg_sent = sum(stats['sentiments']) / len(stats['sentiments'])
            repeat = max(1, int(stats['freq'] * max(SENTIMENT_FLOOR, avg_sent) + 0.5))
            tags.append({
                'word': word,
                'freq': stats['freq'],
                'avg_sentiment': round(avg_sent, 3),
                'repeat_count': repeat
            })
            tag_text_parts.extend([word] * repeat)

        # Sort tags by frequency descending
        tags.sort(key=lambda t: t['freq'], reverse=True)

        first_row = group.iloc[0]
        dishes.append({
            'dish_id': dish_id,
            'dish_name': first_row['dish_name'],
            'restaurant': first_row['restaurant'],
            'tag_text': ' '.join(tag_text_parts),
            'tags': tags,
            'review_count': len(group),
        })

    return pd.DataFrame(dishes)
