import pandas as pd
from app.nlp.keyword import extract_keywords
from app.nlp.sentiment import keyword_sentiment
from app.nlp.matcher import fit_vectorizer, transform_dishes
from app.services.dish_service import build_dish_profiles
from app.services.user_service import build_user_profiles
from app.data.store import (
    save_reviews, save_dishes, save_users, save_vectorizer, save_dish_vectors
)


def run_full_pipeline(csv_path):
    """
    Run the complete pipeline: load CSV -> extract keywords -> build profiles -> vectorize.
    Returns summary dict with counts.
    """
    df = pd.read_csv(csv_path)

    # Step 1: Extract keywords and sentiment per review
    all_keywords = []
    for _, row in df.iterrows():
        kw_list = extract_keywords(row['review_text'])
        kw_words = [w for w, _ in kw_list]
        sentiments = keyword_sentiment(row['review_text'], kw_words)
        merged = [
            {
                'word': w,
                'tfidf_weight': round(weight, 3),
                'sentiment': round(sentiments.get(w, 0.5), 3)
            }
            for w, weight in kw_list
        ]
        all_keywords.append(merged)

    df['keywords'] = all_keywords
    save_reviews(df)

    # Step 2: Build dish profiles
    dishes_df = build_dish_profiles(df)
    save_dishes(dishes_df)

    # Step 3: Build user profiles
    users_df = build_user_profiles(df)
    save_users(users_df)

    # Step 4: Fit vectorizer and transform dishes
    dish_tag_texts = dishes_df['tag_text'].tolist()
    vectorizer = fit_vectorizer(dish_tag_texts)
    dish_vectors = transform_dishes(vectorizer, dish_tag_texts)
    save_vectorizer(vectorizer)
    save_dish_vectors(dish_vectors)

    return {
        'num_reviews': len(df),
        'num_dishes': len(dishes_df),
        'num_users': len(users_df),
        'vocab_size': len(vectorizer.vocabulary_),
    }
