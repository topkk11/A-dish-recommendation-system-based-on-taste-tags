import jieba
import jieba.analyse
from config import STOPWORDS_PATH, KEYWORD_TOP_K, KEYWORD_METHOD


def _load_stopwords():
    with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


STOPWORDS = _load_stopwords()


def segment(text):
    """Tokenize and filter: remove stopwords and single-char tokens."""
    words = jieba.cut(text)
    return [
        w for w in words
        if w not in STOPWORDS
        and len(w) > 1
    ]


def extract_keywords(text, topk=KEYWORD_TOP_K, method=KEYWORD_METHOD):
    """
    Extract top-K keywords from text.
    method: 'tfidf' or 'textrank'
    Returns [(word, weight), ...]
    """
    tokens = segment(text)
    if not tokens:
        return []

    filtered_text = ' '.join(tokens)

    if method == 'textrank':
        return jieba.analyse.textrank(
            filtered_text, topK=topk, withWeight=True
        )
    else:
        return jieba.analyse.extract_tags(
            filtered_text, topK=topk, withWeight=True
        )
