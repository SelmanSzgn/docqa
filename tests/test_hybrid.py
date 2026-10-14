import chromadb
import numpy as np
import pytest

from docqa import pipeline
from docqa.chunking import Chunk
from docqa.store import add_chunks

A = Chunk("the cat sat on the mat", "doc.pdf", 1)
B = Chunk("a quick brown fox jumps", "doc.pdf", 2)
C = Chunk("gpu kernels run parallel threads", "doc.pdf", 3)
D = Chunk("birds fly over the sea", "doc.pdf", 4)
CHUNKS = [A, B, C, D]
VECS = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [-1.0, 0.0]])


def make_collection(tmp_path, fill=True):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("test")
    if fill:
        add_chunks(collection, CHUNKS, VECS)
    return collection


def patch_embed(monkeypatch):
    calls = []

    def fake_embed(model, texts):
        calls.append(texts)
        return np.array([[1.0, 0.0]])

    monkeypatch.setattr(pipeline, "embed_texts", fake_embed)
    return calls


def test_keyword_match_beats_vector_order(tmp_path, monkeypatch):
    patch_embed(monkeypatch)
    collection = make_collection(tmp_path)
    result = pipeline.retrieve_chunks_hybrid("gpu", collection, None, 1)
    assert result == [C]


def test_without_keyword_match_vector_order_is_kept(tmp_path, monkeypatch):
    patch_embed(monkeypatch)
    collection = make_collection(tmp_path)
    result = pipeline.retrieve_chunks_hybrid("zebra", collection, None, 3)
    assert result == [A, C, B]


def test_both_rankings_agreeing_on_first(tmp_path, monkeypatch):
    patch_embed(monkeypatch)
    collection = make_collection(tmp_path)
    result = pipeline.retrieve_chunks_hybrid("cat", collection, None, 2)
    assert result == [A, C]


def test_k_larger_than_collection(tmp_path, monkeypatch):
    patch_embed(monkeypatch)
    collection = make_collection(tmp_path)
    result = pipeline.retrieve_chunks_hybrid("gpu", collection, None, 10)
    assert result == [C, A, B, D]


def test_pool_limits_the_candidates(tmp_path, monkeypatch):
    patch_embed(monkeypatch)
    collection = make_collection(tmp_path)
    result = pipeline.retrieve_chunks_hybrid(
        "gpu", collection, None, 4, pool=1
    )
    assert result == [A, C]


def test_invalid_k_raises_before_any_work(tmp_path, monkeypatch):
    calls = patch_embed(monkeypatch)
    collection = make_collection(tmp_path)
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="k"):
            pipeline.retrieve_chunks_hybrid("gpu", collection, None, bad_k)
    assert calls == []


def test_empty_collection_returns_empty_without_embedding(
    tmp_path, monkeypatch
):
    calls = patch_embed(monkeypatch)
    collection = make_collection(tmp_path, fill=False)
    assert pipeline.retrieve_chunks_hybrid("gpu", collection, None, 3) == []
    assert calls == []


def test_question_is_embedded_as_a_single_text(tmp_path, monkeypatch):
    calls = patch_embed(monkeypatch)
    collection = make_collection(tmp_path)
    pipeline.retrieve_chunks_hybrid("gpu", collection, None, 2)
    assert calls == [["gpu"]]
