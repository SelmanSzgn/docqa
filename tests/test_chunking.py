import pytest

from docqa.chunking import chunk_pages, chunk_text


def test_chunk_text_no_error():
    text = "abcdefghij"
    size = 4
    overlap = 1
    chunks = chunk_text(text, size, overlap)
    assert len(chunks) == 3
    assert chunks[0] == "abcd"
    assert chunks[1] == "defg"
    assert chunks[2] == "ghij"

    assert chunk_text("abcdefghi", 4, 0) == ["abcd", "efgh", "i"]
    assert chunk_text("abcdefghij", 4, 2) == ["abcd", "cdef", "efgh", "ghij"]
    assert chunk_text("a", 4, 1) == ["a"]
    assert chunk_text("ab", 4, 2) == ["ab"]
    assert chunk_text("abcde", 4, 3) == ["abcd", "bcde"]


def test_chunk_text_empty_text():
    assert len(chunk_text("", 4, 1)) == 0


def test_chunk_text_negative_step():
    with pytest.raises(
        ValueError, match="Chunk size value must be greater than overlap."
    ):
        chunk_text("abcdefghij", 1, 3)


def test_chunk_text_negative_overlap():
    with pytest.raises(
        ValueError, match="Overlap value must be non negative."
    ):
        chunk_text("abc", 2, -1)


def test_chunk_text_size_less_or_equal_to_zero():
    with pytest.raises(ValueError, match="Chunk size value must be positive."):
        chunk_text("abc", 0, 1)
    with pytest.raises(ValueError, match="Chunk size value must be positive."):
        chunk_text("abc", -1, 1)


def test_chunk_pages_keeps_page_and_source():
    all_chunks = chunk_pages(
        [(1, "abcdefghij"), (2, "xyz")], source="doc.pdf", size=4, overlap=1
    )
    assert len(all_chunks) == 4
    assert [c.text for c in all_chunks] == ["abcd", "defg", "ghij", "xyz"]
    assert [c.page for c in all_chunks] == [1, 1, 1, 2]
    assert sum([c.source == "doc.pdf" for c in all_chunks]) == 4
