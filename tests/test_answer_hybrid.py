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


def patch_all(monkeypatch):
    seen = {"embed": [], "prompts": []}

    def fake_embed(model, texts):
        seen["embed"].append(texts)
        return np.array([[1.0, 0.0]])

    def fake_generate(prompt):
        seen["prompts"].append(prompt)
        return "final answer"

    monkeypatch.setattr(pipeline, "embed_texts", fake_embed)
    monkeypatch.setattr(pipeline, "generate", fake_generate)
    return seen


def test_keyword_chunk_is_used_in_prompt(tmp_path, monkeypatch):
    seen = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    result = pipeline.answer_hybrid("gpu", collection, None, k=1)
    assert result == "final answer"
    assert len(seen["prompts"]) == 1
    prompt = seen["prompts"][0]
    assert "[doc.pdf, p.3]" in prompt
    assert C.text in prompt
    for other in (A, B, D):
        assert other.text not in prompt
    assert prompt.endswith("Question: gpu")


def test_chunks_appear_in_fused_order(tmp_path, monkeypatch):
    seen = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    pipeline.answer_hybrid("cat", collection, None, k=2)
    prompt = seen["prompts"][0]
    assert A.text in prompt and C.text in prompt
    assert prompt.index(A.text) < prompt.index(C.text)
    assert B.text not in prompt and D.text not in prompt


def test_default_k_is_three(tmp_path, monkeypatch):
    seen = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    pipeline.answer_hybrid("zebra", collection, None)
    prompt = seen["prompts"][0]
    for chunk in (A, C, B):
        assert chunk.text in prompt
    assert D.text not in prompt


def test_k_larger_than_collection(tmp_path, monkeypatch):
    seen = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    pipeline.answer_hybrid("gpu", collection, None, k=10)
    prompt = seen["prompts"][0]
    for chunk in CHUNKS:
        assert chunk.text in prompt


def test_invalid_k_raises_before_any_work(tmp_path, monkeypatch):
    seen = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="k"):
            pipeline.answer_hybrid("gpu", collection, None, k=bad_k)
    assert seen["embed"] == []
    assert seen["prompts"] == []


def test_empty_collection_says_i_dont_know(tmp_path, monkeypatch):
    seen = patch_all(monkeypatch)
    collection = make_collection(tmp_path, fill=False)
    result = pipeline.answer_hybrid("gpu", collection, None, k=3)
    assert result == "I don't know."
    assert seen["embed"] == []
    assert seen["prompts"] == []


def test_question_is_embedded_as_a_single_text(tmp_path, monkeypatch):
    seen = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    pipeline.answer_hybrid("gpu", collection, None, k=2)
    assert seen["embed"] == [["gpu"]]
