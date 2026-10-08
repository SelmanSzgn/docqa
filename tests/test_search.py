import numpy as np
import pytest

from docqa.search import top_k


def test_top_k_normal():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    query = np.array([1.0, 0.0])
    topk = top_k(query, vecs, 2)
    assert topk[0][1] == pytest.approx(1.0)
    assert topk[1][1] == pytest.approx(0.6)
    assert [i for i, _ in topk] == [0, 2]


def test_top_k_large_value_of_k():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    query = np.array([1.0, 0.0])
    topk = top_k(query, vecs, 10)
    assert len(topk) == 3


def test_top_k_negative_k():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    query = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="k"):
        top_k(query, vecs, 0)
