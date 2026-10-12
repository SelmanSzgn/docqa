import pytest

from docqa.lexical import bm25_top_k, tokenize

TEXTS = [
    "the cat sat on the mat",
    "a quick brown fox jumps",
    "the fox and the cat play",
    "parallel computing with gpu",
    "gpu kernels run parallel threads on the gpu",
]


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Hello, World! 28.4 BLEU") == [
        "hello",
        "world",
        "28",
        "4",
        "bleu",
    ]


def test_tokenize_empty_and_symbols_only():
    assert tokenize("") == []
    assert tokenize("  ,, !? ") == []


def test_returns_only_texts_containing_a_query_word():
    assert bm25_top_k("mat", TEXTS, 3) == [0]


def test_best_match_comes_first():
    result = bm25_top_k("fox cat", TEXTS, 3)
    assert result[0] == 2
    assert set(result) == {0, 1, 2}


def test_k_limits_results():
    assert bm25_top_k("fox cat", TEXTS, 1) == [2]


def test_k_larger_than_matches():
    result = bm25_top_k("fox cat", TEXTS, 10)
    assert len(result) == 3
    assert set(result) == {0, 1, 2}


def test_no_match_returns_empty():
    assert bm25_top_k("zebra", TEXTS, 3) == []


def test_query_is_case_and_punctuation_insensitive():
    assert bm25_top_k("  MAT? ", TEXTS, 3) == [0]
    assert bm25_top_k("FOX, Cat!", TEXTS, 3)[0] == 2


def test_empty_query_returns_empty():
    assert bm25_top_k("", TEXTS, 3) == []
    assert bm25_top_k("  ?! ", TEXTS, 3) == []


def test_empty_texts_returns_empty():
    assert bm25_top_k("cat", [], 3) == []


def test_texts_without_any_token_return_empty():
    assert bm25_top_k("cat", ["", "   ", "..."], 3) == []


def test_blank_text_among_others_keeps_original_indices():
    texts = [
        "",
        "the cat sat on the mat",
        "a quick brown fox jumps",
        "dogs run",
        "birds fly",
    ]
    assert bm25_top_k("mat", texts, 3) == [1]


def test_invalid_k():
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="k"):
            bm25_top_k("cat", TEXTS, bad_k)
        with pytest.raises(ValueError, match="k"):
            bm25_top_k("cat", [], bad_k)
