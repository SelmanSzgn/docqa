import json
from pathlib import Path

import pytest

from docqa.evaluation import load_dataset

REAL_DATASET = (
    Path(__file__).parent.parent / "eval" / "attention_questions.json"
)


def write(tmp_path, content):
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_valid_dataset_is_returned_unchanged(tmp_path):
    content = [
        {"question": "q1", "relevant_pages": [5]},
        {"question": "q2", "relevant_pages": [7, 9]},
    ]
    assert load_dataset(write(tmp_path, content)) == content


def test_empty_list_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="dataset"):
        load_dataset(write(tmp_path, []))


def test_non_list_content_is_rejected(tmp_path):
    content = {"question": "q", "relevant_pages": [1]}
    with pytest.raises(ValueError, match="dataset"):
        load_dataset(write(tmp_path, content))


def test_missing_question_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="question"):
        load_dataset(write(tmp_path, [{"relevant_pages": [1]}]))


def test_blank_or_non_string_question_is_rejected(tmp_path):
    for bad in ("", "   ", 42):
        content = [{"question": bad, "relevant_pages": [1]}]
        with pytest.raises(ValueError, match="question"):
            load_dataset(write(tmp_path, content))


def test_missing_relevant_pages_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="relevant_pages"):
        load_dataset(write(tmp_path, [{"question": "q"}]))


def test_empty_relevant_pages_is_rejected(tmp_path):
    content = [{"question": "q", "relevant_pages": []}]
    with pytest.raises(ValueError, match="relevant_pages"):
        load_dataset(write(tmp_path, content))


def test_relevant_pages_not_a_list_is_rejected(tmp_path):
    content = [{"question": "q", "relevant_pages": 5}]
    with pytest.raises(ValueError, match="relevant_pages"):
        load_dataset(write(tmp_path, content))


def test_invalid_page_values_are_rejected(tmp_path):
    for bad in (0, -1, "5", 2.5):
        content = [{"question": "q", "relevant_pages": [3, bad]}]
        with pytest.raises(ValueError, match="relevant_pages"):
            load_dataset(write(tmp_path, content))


def test_real_dataset_file_is_valid():
    dataset = load_dataset(REAL_DATASET)
    questions = [item["question"] for item in dataset]
    assert len(dataset) >= 15
    assert len(set(questions)) == len(questions)
    assert all(p <= 15 for item in dataset for p in item["relevant_pages"])
