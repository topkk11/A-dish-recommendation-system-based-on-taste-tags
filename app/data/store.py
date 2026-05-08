import pickle
import os
import pandas as pd
from scipy.sparse import load_npz, save_npz
from config import REVIEWS_PATH, DISHES_PATH, USERS_PATH, VECTORIZER_PATH, DISH_VECTORS_PATH


def _ensure_output_dir():
    os.makedirs(os.path.dirname(REVIEWS_PATH), exist_ok=True)


def save_reviews(df):
    _ensure_output_dir()
    with open(REVIEWS_PATH, 'wb') as f:
        pickle.dump(df, f)


def load_reviews():
    with open(REVIEWS_PATH, 'rb') as f:
        return pickle.load(f)


def save_dishes(df):
    _ensure_output_dir()
    with open(DISHES_PATH, 'wb') as f:
        pickle.dump(df, f)


def load_dishes():
    with open(DISHES_PATH, 'rb') as f:
        return pickle.load(f)


def save_users(df):
    _ensure_output_dir()
    with open(USERS_PATH, 'wb') as f:
        pickle.dump(df, f)


def load_users():
    with open(USERS_PATH, 'rb') as f:
        return pickle.load(f)


def save_vectorizer(vectorizer):
    _ensure_output_dir()
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)


def load_vectorizer():
    with open(VECTORIZER_PATH, 'rb') as f:
        return pickle.load(f)


def save_dish_vectors(matrix):
    _ensure_output_dir()
    save_npz(DISH_VECTORS_PATH, matrix)


def load_dish_vectors():
    return load_npz(DISH_VECTORS_PATH)


def all_outputs_exist():
    return all(
        os.path.exists(p)
        for p in [REVIEWS_PATH, DISHES_PATH, USERS_PATH, VECTORIZER_PATH, DISH_VECTORS_PATH]
    )
