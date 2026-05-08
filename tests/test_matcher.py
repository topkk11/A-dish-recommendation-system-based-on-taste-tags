from app.nlp.matcher import fit_vectorizer, transform_dishes, transform_user, top_n_similar
from app.nlp.keyword import extract_keywords
from app.nlp.sentiment import keyword_sentiment
from app.services.dish_service import build_dish_profiles
from app.services.user_service import build_user_profiles


def _build_test_data():
    """Build a mini reviews dataset for testing the matcher."""
    import pandas as pd

    data = [
        {'review_id': 1, 'user_id': 'U1', 'dish_id': 'D1', 'dish_name': '麻辣火锅',
         'restaurant': '川味轩', 'review_text': '麻辣鲜香非常过瘾', 'rating': 5},
        {'review_id': 2, 'user_id': 'U1', 'dish_id': 'D2', 'dish_name': '清蒸鱼',
         'restaurant': '粤菜馆', 'review_text': '清淡鲜美鱼肉很嫩', 'rating': 5},
        {'review_id': 3, 'user_id': 'U2', 'dish_id': 'D1', 'dish_name': '麻辣火锅',
         'restaurant': '川味轩', 'review_text': '太辣了受不了', 'rating': 2},
        {'review_id': 4, 'user_id': 'U2', 'dish_id': 'D2', 'dish_name': '清蒸鱼',
         'restaurant': '粤菜馆', 'review_text': '清淡好吃很健康', 'rating': 5},
    ]
    df = pd.DataFrame(data)

    all_keywords = []
    for _, row in df.iterrows():
        kw_list = extract_keywords(row['review_text'])
        kw_words = [w for w, _ in kw_list]
        sentiments = keyword_sentiment(row['review_text'], kw_words)
        merged = [{'word': w, 'tfidf_weight': round(wt, 3), 'sentiment': round(sentiments.get(w, 0.5), 3)}
                  for w, wt in kw_list]
        all_keywords.append(merged)
    df['keywords'] = all_keywords
    return df


def test_fit_vectorizer():
    df = _build_test_data()
    dishes_df = build_dish_profiles(df)
    tag_texts = dishes_df['tag_text'].tolist()
    vec = fit_vectorizer(tag_texts)
    assert vec is not None
    assert len(vec.vocabulary_) > 0


def test_transform_dishes():
    df = _build_test_data()
    dishes_df = build_dish_profiles(df)
    tag_texts = dishes_df['tag_text'].tolist()
    vec = fit_vectorizer(tag_texts)
    matrix = transform_dishes(vec, tag_texts)
    assert matrix.shape[0] == len(dishes_df)


def test_transform_user():
    df = _build_test_data()
    dishes_df = build_dish_profiles(df)
    users_df = build_user_profiles(df)
    tag_texts = dishes_df['tag_text'].tolist()
    vec = fit_vectorizer(tag_texts)
    dish_matrix = transform_dishes(vec, tag_texts)

    u1 = users_df[users_df['user_id'] == 'U1'].iloc[0]
    u1_vec = transform_user(vec, u1['preference_text'])
    assert u1_vec.shape[1] == dish_matrix.shape[1]


def test_top_n_similar():
    df = _build_test_data()
    dishes_df = build_dish_profiles(df)
    users_df = build_user_profiles(df)
    tag_texts = dishes_df['tag_text'].tolist()
    vec = fit_vectorizer(tag_texts)
    dish_matrix = transform_dishes(vec, tag_texts)
    dish_ids = dishes_df['dish_id'].tolist()

    # U1 likes everything, should rank both high
    u1 = users_df[users_df['user_id'] == 'U1'].iloc[0]
    u1_vec = transform_user(vec, u1['preference_text'])
    results = top_n_similar(u1_vec, dish_matrix, dish_ids, n=2)
    assert len(results) == 2
    assert results[0][1] >= results[1][1]  # descending


def test_similarity_range():
    df = _build_test_data()
    dishes_df = build_dish_profiles(df)
    users_df = build_user_profiles(df)
    tag_texts = dishes_df['tag_text'].tolist()
    vec = fit_vectorizer(tag_texts)
    dish_matrix = transform_dishes(vec, tag_texts)
    dish_ids = dishes_df['dish_id'].tolist()

    for _, user_row in users_df.iterrows():
        u_vec = transform_user(vec, user_row['preference_text'])
        results = top_n_similar(u_vec, dish_matrix, dish_ids, n=5)
        for _, score in results:
            assert 0.0 <= score <= 1.0
