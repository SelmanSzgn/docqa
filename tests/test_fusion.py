import pytest

from docqa.fusion import rrf_fuse


def test_symmetric_rankings_tie_broken_by_id():
    assert rrf_fuse([["a", "b", "c"], ["c", "b", "a"]], 3) == ["a", "c", "b"]


def test_item_in_both_rankings_beats_item_in_one():
    assert rrf_fuse([["a", "b"], ["c", "b"]], 3) == ["b", "a", "c"]


def test_top_n_limits_results():
    assert rrf_fuse([["a", "b"], ["c", "b"]], 1) == ["b"]


def test_top_n_larger_than_candidates():
    assert rrf_fuse([["a", "b"], ["c", "b"]], 10) == ["b", "a", "c"]


def test_single_ranking_is_unchanged():
    assert rrf_fuse([["e", "c", "i"]], 3) == ["e", "c", "i"]


def test_empty_inputs():
    assert rrf_fuse([], 3) == []
    assert rrf_fuse([[], []], 3) == []


def test_equal_scores_are_ordered_by_id():
    assert rrf_fuse([["a"], ["b"]], 2) == ["a", "b"]
    assert rrf_fuse([["b"], ["a"]], 2) == ["a", "b"]


def test_constant_c_changes_the_result():
    assert rrf_fuse([["a", "b"], ["c", "b"]], 3, c=0) == ["a", "b", "c"]


def test_rank_position_matters():
    rankings = [["w", "x", "y", "z"], ["z", "y"]]
    assert rrf_fuse(rankings, 4) == ["z", "y", "w", "x"]


def test_order_of_rankings_does_not_matter():
    first = [["w", "x", "y", "z"], ["z", "y"]]
    second = [["z", "y"], ["w", "x", "y", "z"]]
    assert rrf_fuse(first, 4) == rrf_fuse(second, 4)


def test_invalid_top_n():
    for bad in (0, -1):
        with pytest.raises(ValueError, match="top_n"):
            rrf_fuse([["a"]], bad)
        with pytest.raises(ValueError, match="top_n"):
            rrf_fuse([], bad)
