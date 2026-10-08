import chromadb
import numpy as np
import pytest

from docqa import pipeline
from docqa.chunking import Chunk
from docqa.store import add_chunks

CHUNKS = [
    Chunk("alpha", "doc.pdf", 1),
    Chunk("beta", "doc.pdf", 2),
    Chunk("gamma", "doc.pdf", 3),
]
VECS = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])


def make_collection(path):
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection("test")
    add_chunks(collection, CHUNKS, VECS)
    return collection


def patch_pipeline(monkeypatch):
    captured = {}

    def fake_embed(model, texts):
        captured["texts"] = texts
        return np.array([[1.0, 0.0]])

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return "final answer"

    monkeypatch.setattr(pipeline, "embed_texts", fake_embed)
    monkeypatch.setattr(pipeline, "generate", fake_generate)
    return captured


def test_answer_from_store_uses_best_chunks(tmp_path, monkeypatch):
    captured = patch_pipeline(monkeypatch)
    collection = make_collection(tmp_path)
    result = pipeline.answer_from_store("What?", collection, None, k=2)
    assert result == "final answer"
    assert captured["texts"] == ["What?"]
    assert "alpha" in captured["prompt"]
    assert "gamma" in captured["prompt"]
    assert "beta" not in captured["prompt"]
    assert captured["prompt"].endswith("Question: What?")


def test_answer_from_store_k_larger_than_collection(tmp_path, monkeypatch):
    captured = patch_pipeline(monkeypatch)
    collection = make_collection(tmp_path)
    pipeline.answer_from_store("What?", collection, None, k=10)
    for word in ("alpha", "beta", "gamma"):
        assert word in captured["prompt"]


def test_answer_from_store_rejects_invalid_k(tmp_path, monkeypatch):
    captured = patch_pipeline(monkeypatch)
    collection = make_collection(tmp_path)
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="k"):
            pipeline.answer_from_store("What?", collection, None, k=bad_k)
    assert "prompt" not in captured
