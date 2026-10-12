import pytest

from docqa.evaluation import hit_at_k, reciprocal_rank


def test_hit_found_within_k():
    assert hit_at_k([3, 5, 7], {5}, 2) == 1.0


def test_hit_found_beyond_k():
    assert hit_at_k([3, 5, 7], {7}, 2) == 0.0


def test_hit_k_larger_than_list():
    assert hit_at_k([3, 5], {5}, 10) == 1.0


def test_hit_empty_retrieved():
    assert hit_at_k([], {5}, 3) == 0.0


def test_hit_several_relevant_pages():
    assert hit_at_k([3, 5, 7], {7, 9}, 3) == 1.0


def test_hit_invalid_k():
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="k"):
            hit_at_k([3, 5], {5}, bad_k)


def test_hit_empty_relevant():
    with pytest.raises(ValueError, match="relevant"):
        hit_at_k([3, 5], set(), 2)


def test_rr_first_position():
    assert reciprocal_rank([5, 3, 7], {5}) == 1.0


def test_rr_third_position():
    assert reciprocal_rank([3, 7, 5], {5}) == pytest.approx(1 / 3)


def test_rr_not_found():
    assert reciprocal_rank([3, 7], {5}) == 0.0


def test_rr_empty_retrieved():
    assert reciprocal_rank([], {5}) == 0.0


def test_rr_duplicates_are_not_removed():
    assert reciprocal_rank([5, 5, 3], {3}) == pytest.approx(1 / 3)


def test_rr_first_of_several_relevant_pages():
    assert reciprocal_rank([3, 5, 7], {7, 5}) == 0.5


def test_rr_empty_relevant():
    with pytest.raises(ValueError, match="relevant"):
        reciprocal_rank([3, 5], set())
