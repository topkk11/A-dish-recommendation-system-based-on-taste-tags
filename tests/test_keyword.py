from app.nlp.keyword import segment, extract_keywords


def test_segment_returns_tokens():
    result = segment('鸡肉很嫩麻辣味刚好')
    assert len(result) > 0
    assert all(len(w) > 1 for w in result)


def test_segment_filters_stopwords():
    result = segment('这个菜很好吃')
    assert '这个' not in result
    assert '的' not in result


def test_segment_removes_single_chars():
    result = segment('菜很香')
    assert '菜' not in result
    assert '很' not in result


def test_extract_keywords_tfidf(positive_review):
    result = extract_keywords(positive_review, topk=3, method='tfidf')
    assert len(result) <= 3
    for word, weight in result:
        assert isinstance(word, str)
        assert weight > 0


def test_extract_keywords_textrank(positive_review):
    result = extract_keywords(positive_review, topk=3, method='textrank')
    # textrank may return fewer results on short text
    for word, weight in result:
        assert isinstance(word, str)


def test_extract_keywords_negative(negative_review):
    result = extract_keywords(negative_review, topk=5, method='tfidf')
    assert len(result) > 0
