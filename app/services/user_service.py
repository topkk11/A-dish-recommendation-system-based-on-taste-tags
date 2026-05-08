from collections import defaultdict
import pandas as pd
from config import SENTIMENT_FLOOR


def build_user_profiles(reviews_df):
    """
    Aggregate reviews by user_id to build user preference profiles.
    Returns DataFrame with columns: user_id, preference_text, preference_tags, review_count
    """
    users = []

    for user_id, group in reviews_df.groupby('user_id'):
        keyword_stats = defaultdict(lambda: {'freq': 0, 'sentiments': []})

        for _, row in group.iterrows():
            for kw in row['keywords']:
                word = kw['word']
                keyword_stats[word]['freq'] += 1
                keyword_stats[word]['sentiments'].append(kw['sentiment'])

        tags = []
        tag_text_parts = []
        for word, stats in keyword_stats.items():
            avg_sent = sum(stats['sentiments']) / len(stats['sentiments'])
            weight = stats['freq'] * max(SENTIMENT_FLOOR, avg_sent)
            repeat = max(1, int(weight + 0.5))
            tags.append({
                'word': word,
                'weight': round(weight, 2),
                'freq': stats['freq'],
                'avg_sentiment': round(avg_sent, 3),
                'repeat_count': repeat
            })
            tag_text_parts.extend([word] * repeat)

        tags.sort(key=lambda t: t['weight'], reverse=True)

        users.append({
            'user_id': user_id,
            'preference_text': ' '.join(tag_text_parts),
            'preference_tags': tags,
            'review_count': len(group),
        })

    return pd.DataFrame(users)
