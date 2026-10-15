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
SCORES = {A.text: 0.1, B.text: 0.5, C.text: 0.3, D.text: 0.9}


def make_collection(tmp_path, fill=True):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("test")
    if fill:
        add_chunks(collection, CHUNKS, VECS)
    return collection


def patch_all(monkeypatch):
    seen = {"embed": [], "prompts": [], "pairs": []}

    def fake_embed(model, texts):
        seen["embed"].append(texts)
        return np.array([[1.0, 0.0]])

    def fake_generate(prompt):
        seen["prompts"].append(prompt)
        return "final answer"

    def scorer(pairs):
        seen["pairs"].append(list(pairs))
        return [SCORES[text] for _, text in pairs]

    monkeypatch.setattr(pipeline, "embed_texts", fake_embed)
    monkeypatch.setattr(pipeline, "generate", fake_generate)
    return seen, scorer


def test_reranker_overrides_retrieval_order(tmp_path, monkeypatch):
    seen, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    result = pipeline.answer_reranked(
        "zebra", collection, None, scorer, k=2, pool=4
    )
    assert result == "final answer"
    prompt = seen["prompts"][0]
    assert D.text in prompt and B.text in prompt
    assert prompt.index(D.text) < prompt.index(B.text)
    assert A.text not in prompt and C.text not in prompt
    assert prompt.endswith("Question: zebra")


def test_defaults_are_k_three_and_pool_twenty(tmp_path, monkeypatch):
    seen, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    pipeline.answer_reranked("zebra", collection, None, scorer)
    prompt = seen["prompts"][0]
    for chunk in (D, B, C):
        assert chunk.text in prompt
    assert prompt.index(D.text) < prompt.index(B.text) < prompt.index(C.text)
    assert A.text not in prompt
    assert len(seen["pairs"][0]) == 4


def test_pool_limits_candidates_seen_by_scorer(tmp_path, monkeypatch):
    seen, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    pipeline.answer_reranked("zebra", collection, None, scorer, k=2, pool=2)
    assert seen["pairs"] == [[("zebra", A.text), ("zebra", C.text)]]
    prompt = seen["prompts"][0]
    assert A.text in prompt and C.text in prompt
    assert D.text not in prompt and B.text not in prompt


def test_k_larger_than_candidates(tmp_path, monkeypatch):
    seen, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    pipeline.answer_reranked("zebra", collection, None, scorer, k=10, pool=10)
    prompt = seen["prompts"][0]
    for chunk in CHUNKS:
        assert chunk.text in prompt


def test_scorer_called_once_and_k_limits_prompt(tmp_path, monkeypatch):
    seen, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    pipeline.answer_reranked("zebra", collection, None, scorer, k=1, pool=4)
    assert len(seen["pairs"]) == 1
    prompt = seen["prompts"][0]
    assert D.text in prompt
    for other in (A, B, C):
        assert other.text not in prompt


def test_invalid_k_raises_before_any_work(tmp_path, monkeypatch):
    seen, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="k"):
            pipeline.answer_reranked(
                "zebra", collection, None, scorer, k=bad_k
            )
    assert seen["embed"] == [] and seen["pairs"] == []
    assert seen["prompts"] == []


def test_pool_smaller_than_k_raises_before_any_work(tmp_path, monkeypatch):
    seen, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    with pytest.raises(ValueError, match="pool"):
        pipeline.answer_reranked(
            "zebra", collection, None, scorer, k=3, pool=2
        )
    assert seen["embed"] == [] and seen["pairs"] == []
    assert seen["prompts"] == []


def test_empty_collection_says_i_dont_know(tmp_path, monkeypatch):
    seen, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path, fill=False)
    result = pipeline.answer_reranked("zebra", collection, None, scorer)
    assert result == "I don't know."
    assert seen["embed"] == [] and seen["pairs"] == []
    assert seen["prompts"] == []


def test_question_is_embedded_as_a_single_text(tmp_path, monkeypatch):
    seen, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    pipeline.answer_reranked("zebra", collection, None, scorer)
    assert seen["embed"] == [["zebra"]]
