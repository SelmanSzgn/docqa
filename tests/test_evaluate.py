import pytest

from docqa.evaluation import evaluate


def make_retrieve(answers):
    calls = []

    def retrieve(question):
        calls.append(question)
        return answers[question]

    return retrieve, calls


DATASET = [
    {"question": "q1", "relevant_pages": [5]},
    {"question": "q2", "relevant_pages": [7, 9]},
]


def test_perfect_retrieval():
    retrieve, _ = make_retrieve({"q1": [5, 1], "q2": [9, 2]})
    result = evaluate(retrieve, DATASET, k=2)
    assert set(result) == {"hit_rate", "mrr"}
    assert result["hit_rate"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(1.0)


def test_mixed_retrieval():
    retrieve, _ = make_retrieve({"q1": [3, 5, 7], "q2": [1, 2, 3]})
    result = evaluate(retrieve, DATASET, k=2)
    assert result["hit_rate"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.25)


def test_k_limits_hit_rate_but_not_mrr():
    retrieve, _ = make_retrieve({"q1": [1, 2, 5], "q2": [7]})
    result = evaluate(retrieve, DATASET, k=2)
    assert result["hit_rate"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(2 / 3)


def test_retrieve_called_once_per_question_in_order():
    retrieve, calls = make_retrieve({"q1": [5], "q2": [7]})
    evaluate(retrieve, DATASET, k=1)
    assert calls == ["q1", "q2"]


def test_empty_dataset():
    retrieve, _ = make_retrieve({})
    with pytest.raises(ValueError, match="dataset"):
        evaluate(retrieve, [], k=2)


def test_invalid_k():
    retrieve, calls = make_retrieve({"q1": [5], "q2": [7]})
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="k"):
            evaluate(retrieve, DATASET, k=bad_k)
    assert calls == []


def test_empty_relevant_pages():
    retrieve, _ = make_retrieve({"q1": [5]})
    dataset = [{"question": "q1", "relevant_pages": []}]
    with pytest.raises(ValueError, match="relevant"):
        evaluate(retrieve, dataset, k=2)
