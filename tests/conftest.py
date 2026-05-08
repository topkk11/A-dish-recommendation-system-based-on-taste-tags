import pytest
import pandas as pd
from config import SAMPLE_REVIEWS_PATH


@pytest.fixture
def sample_df():
    return pd.read_csv(SAMPLE_REVIEWS_PATH)


@pytest.fixture
def positive_review():
    return '鸡肉很嫩花生很脆麻辣味刚好超级下饭'


@pytest.fixture
def negative_review():
    return '太辣了受不了花椒味太重不会再点'
