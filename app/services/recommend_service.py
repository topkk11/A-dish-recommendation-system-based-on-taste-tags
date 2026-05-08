from app.data.store import load_users, load_dishes, load_vectorizer, load_dish_vectors
from app.nlp.matcher import transform_user, top_n_similar


def recommend(user_id, top_n=15):
    """
    Generate dish recommendations for a user.
    Returns list of dicts with dish info, similarity score, and matching tags.
    """
    users_df = load_users()
    dishes_df = load_dishes()
    vectorizer = load_vectorizer()
    dish_vectors = load_dish_vectors()

    user_row = users_df[users_df['user_id'] == user_id].iloc[0]
    preference_text = user_row['preference_text']
    user_pref_keywords = {t['word'] for t in user_row['preference_tags']}

    user_vec = transform_user(vectorizer, preference_text)
    dish_ids = dishes_df['dish_id'].tolist()
    top_dishes = top_n_similar(user_vec, dish_vectors, dish_ids, n=top_n)

    results = []
    for dish_id, score in top_dishes:
        dish_row = dishes_df[dishes_df['dish_id'] == dish_id].iloc[0]
        dish_keywords = {t['word'] for t in dish_row['tags']}
        matching_tags = list(user_pref_keywords & dish_keywords)

        results.append({
            'dish_id': dish_id,
            'dish_name': dish_row['dish_name'],
            'restaurant': dish_row['restaurant'],
            'score': round(score, 4),
            'matching_tags': matching_tags,
            'review_count': dish_row['review_count'],
        })

    return results
