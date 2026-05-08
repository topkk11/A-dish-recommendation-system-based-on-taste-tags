import re
from snownlp import SnowNLP


def analyze(text):
    """Returns sentiment score 0 (negative) to 1 (positive)."""
    return SnowNLP(text).sentiments


def keyword_sentiment(review_text, keywords):
    """
    For each keyword, find the sentence containing it and score sentiment.
    Returns {keyword: sentiment_score}.
    """
    sentences = re.split(r'[。！？，、；：\n]+', review_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    result = {}
    for kw in keywords:
        containing = [s for s in sentences if kw in s]
        if containing:
            result[kw] = analyze(containing[0])
        else:
            result[kw] = 0.5
    return result
