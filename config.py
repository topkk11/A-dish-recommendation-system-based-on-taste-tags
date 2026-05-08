import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

STOPWORDS_PATH = os.path.join(DATA_DIR, 'stopwords.txt')
SAMPLE_REVIEWS_PATH = os.path.join(DATA_DIR, 'sample_reviews.csv')

REVIEWS_PATH = os.path.join(OUTPUT_DIR, 'reviews.p')
DISHES_PATH = os.path.join(OUTPUT_DIR, 'dishes.p')
USERS_PATH = os.path.join(OUTPUT_DIR, 'users.p')
VECTORIZER_PATH = os.path.join(OUTPUT_DIR, 'vectorizer.p')
DISH_VECTORS_PATH = os.path.join(OUTPUT_DIR, 'dish_vectors.npz')

KEYWORD_TOP_K = 10
KEYWORD_METHOD = 'tfidf'  # 'tfidf' or 'textrank'
SENTIMENT_FLOOR = 0.3     # minimum weight multiplier for negative keywords
RECOMMEND_TOP_N = 15
