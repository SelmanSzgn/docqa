import numpy as np
import pytest

from docqa.chunking import Chunk
from docqa.reranker import rerank

A = Chunk("alpha", "doc.pdf", 1)
B = Chunk("beta", "doc.pdf", 2)
C = Chunk("gamma", "doc.pdf", 3)


def make_scorer(scores_by_text, as_array=False):
    calls = []

    def scorer(pairs):
        calls.append(list(pairs))
        scores = [scores_by_text[text] for _, text in pairs]
        return np.array(scores) if as_array else scores

    return scorer, calls


SCORES = {"alpha": 0.1, "beta": 0.9, "gamma": 0.5}


def test_orders_by_score_descending():
    scorer, _ = make_scorer(SCORES)
    assert rerank("What?", [A, B, C], scorer, 3) == [B, C, A]


def test_k_limits_results():
    scorer, _ = make_scorer(SCORES)
    assert rerank("What?", [A, B, C], scorer, 1) == [B]


def test_k_larger_than_chunks():
    scorer, _ = make_scorer(SCORES)
    assert rerank("What?", [A, B, C], scorer, 10) == [B, C, A]


def test_ties_keep_original_order():
    scorer, _ = make_scorer({"alpha": 0.5, "beta": 0.5, "gamma": 0.5})
    assert rerank("What?", [A, B, C], scorer, 3) == [A, B, C]


def test_empty_chunks_do_not_call_scorer():
    scorer, calls = make_scorer(SCORES)
    assert rerank("What?", [], scorer, 3) == []
    assert calls == []


def test_invalid_k_raises_before_scoring():
    scorer, calls = make_scorer(SCORES)
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="k"):
            rerank("What?", [A, B, C], scorer, bad_k)
    assert calls == []


def test_scorer_called_once_with_question_text_pairs():
    scorer, calls = make_scorer(SCORES)
    rerank("What?", [A, B, C], scorer, 2)
    assert calls == [
        [("What?", "alpha"), ("What?", "beta"), ("What?", "gamma")]
    ]


def test_numpy_scores_are_accepted():
    scorer, _ = make_scorer(SCORES, as_array=True)
    assert rerank("What?", [A, B, C], scorer, 2) == [B, C]
