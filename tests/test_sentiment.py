from app.nlp.sentiment import analyze, keyword_sentiment


def test_analyze_positive():
    score = analyze('鸡肉很嫩超级好吃非常香')
    assert score > 0.5


def test_analyze_negative():
    score = analyze('太辣了受不了不会再点太难吃')
    assert score < 0.5


def test_analyze_returns_zero_to_one():
    score = analyze('一般般没什么特别的')
    assert 0.0 <= score <= 1.0


def test_keyword_sentiment_positive_text(positive_review):
    keywords = ['嫩', '脆', '麻辣']
    scores = keyword_sentiment(positive_review, keywords)
    assert set(scores.keys()) == set(keywords)
    for s in scores.values():
        assert 0.0 <= s <= 1.0


def test_keyword_sentiment_missing_keyword():
    scores = keyword_sentiment('这个菜很好吃', ['不存在的词'])
    assert '不存在的词' in scores
    assert scores['不存在的词'] == 0.5
