import chromadb
import numpy as np
import pytest

from docqa import pipeline
from docqa.chunking import Chunk
from docqa.store import add_chunks

CHUNKS = [
    Chunk("alpha", "doc.pdf", 4),
    Chunk("beta", "doc.pdf", 7),
    Chunk("gamma", "doc.pdf", 4),
]
VECS = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])


def make_collection(path):
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection("test")
    add_chunks(collection, CHUNKS, VECS)
    return collection


def patch_embed(monkeypatch, vector):
    calls = []

    def fake_embed(model, texts):
        calls.append(texts)
        return np.array([vector])

    monkeypatch.setattr(pipeline, "embed_texts", fake_embed)
    return calls


def test_pages_are_ordered_closest_first(tmp_path, monkeypatch):
    patch_embed(monkeypatch, [1.0, 0.0])
    collection = make_collection(tmp_path)
    assert pipeline.retrieve_pages("What?", collection, None, 3) == [4, 4, 7]


def test_other_query_gives_other_order(tmp_path, monkeypatch):
    patch_embed(monkeypatch, [0.0, 1.0])
    collection = make_collection(tmp_path)
    assert pipeline.retrieve_pages("What?", collection, None, 3) == [7, 4, 4]


def test_only_k_pages_are_returned(tmp_path, monkeypatch):
    patch_embed(monkeypatch, [1.0, 0.0])
    collection = make_collection(tmp_path)
    assert pipeline.retrieve_pages("What?", collection, None, 1) == [4]


def test_k_larger_than_collection(tmp_path, monkeypatch):
    patch_embed(monkeypatch, [1.0, 0.0])
    collection = make_collection(tmp_path)
    assert len(pipeline.retrieve_pages("What?", collection, None, 10)) == 3


def test_invalid_k_raises_before_embedding(tmp_path, monkeypatch):
    calls = patch_embed(monkeypatch, [1.0, 0.0])
    collection = make_collection(tmp_path)
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="k"):
            pipeline.retrieve_pages("What?", collection, None, bad_k)
    assert calls == []


def test_question_is_embedded_as_a_single_text(tmp_path, monkeypatch):
    calls = patch_embed(monkeypatch, [1.0, 0.0])
    collection = make_collection(tmp_path)
    pipeline.retrieve_pages("What?", collection, None, 2)
    assert calls == [["What?"]]
