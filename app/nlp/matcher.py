from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def _identity_tokenizer(text):
    """Split pre-tokenized whitespace-separated text into tokens."""
    return text.split()


def fit_vectorizer(dish_tag_texts):
    """
    Fit TF-IDF vectorizer on dish tag texts.
    Uses unigrams + bigrams for Chinese compound flavor descriptors.
    """
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        tokenizer=_identity_tokenizer,
        lowercase=False,
    )
    vectorizer.fit(dish_tag_texts)
    return vectorizer


def transform_dishes(vectorizer, dish_tag_texts):
    """Transform dish tag texts to TF-IDF matrix."""
    return vectorizer.transform(dish_tag_texts)


def transform_user(vectorizer, preference_text):
    """Transform one user's preference text to TF-IDF vector."""
    return vectorizer.transform([preference_text])


def top_n_similar(user_vector, dish_matrix, dish_ids, n=10):
    """
    Return top-N (dish_id, similarity_score) sorted descending.
    """
    scores = cosine_similarity(user_vector, dish_matrix).flatten()
    top_indices = scores.argsort()[::-1][:n]
    return [(dish_ids[i], float(scores[i])) for i in top_indices]
