import json

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

P = Chunk("alpha one", "doc.pdf", 1)
Q = Chunk("alpha two", "doc.pdf", 1)
R = Chunk("beta", "other.pdf", 1)
DUP_CHUNKS = [P, Q, R]
DUP_VECS = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])

SCORES = {
    A.text: 0.1,
    B.text: 0.5,
    C.text: 0.3,
    D.text: 0.9,
    P.text: 0.9,
    R.text: 0.8,
    Q.text: 0.7,
}


def make_collection(tmp_path, chunks=CHUNKS, vecs=VECS, fill=True):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("test")
    if fill:
        add_chunks(collection, chunks, vecs)
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


def test_answer_and_sources_follow_reranked_order(tmp_path, monkeypatch):
    seen, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    result = pipeline.answer_with_sources(
        "zebra", collection, None, scorer, k=2, pool=4
    )
    assert result == {
        "answer": "final answer",
        "sources": [
            {"source": "doc.pdf", "page": 4},
            {"source": "doc.pdf", "page": 2},
        ],
    }
    prompt = seen["prompts"][0]
    assert D.text in prompt and B.text in prompt
    assert A.text not in prompt and C.text not in prompt


def test_duplicate_source_page_is_listed_once(tmp_path, monkeypatch):
    _, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path, DUP_CHUNKS, DUP_VECS)
    result = pipeline.answer_with_sources(
        "zebra", collection, None, scorer, k=3, pool=3
    )
    assert result["sources"] == [
        {"source": "doc.pdf", "page": 1},
        {"source": "other.pdf", "page": 1},
    ]


def test_result_has_exactly_two_keys_and_is_json_serializable(
    tmp_path, monkeypatch
):
    _, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    result = pipeline.answer_with_sources("zebra", collection, None, scorer)
    assert set(result) == {"answer", "sources"}
    assert json.loads(json.dumps(result)) == result
    assert all(isinstance(s["page"], int) for s in result["sources"])


def test_k_larger_than_candidates_lists_every_page(tmp_path, monkeypatch):
    _, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    result = pipeline.answer_with_sources(
        "zebra", collection, None, scorer, k=10, pool=10
    )
    assert [s["page"] for s in result["sources"]] == [4, 2, 3, 1]


def test_scorer_and_generate_are_called_once(tmp_path, monkeypatch):
    seen, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    pipeline.answer_with_sources("zebra", collection, None, scorer)
    assert len(seen["pairs"]) == 1
    assert len(seen["prompts"]) == 1
    assert seen["embed"] == [["zebra"]]


def test_invalid_k_raises_before_any_work(tmp_path, monkeypatch):
    seen, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="k"):
            pipeline.answer_with_sources(
                "zebra", collection, None, scorer, k=bad_k
            )
    assert seen["embed"] == [] and seen["pairs"] == []
    assert seen["prompts"] == []


def test_pool_smaller_than_k_raises_before_any_work(tmp_path, monkeypatch):
    seen, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path)
    with pytest.raises(ValueError, match="pool"):
        pipeline.answer_with_sources(
            "zebra", collection, None, scorer, k=3, pool=2
        )
    assert seen["embed"] == [] and seen["pairs"] == []
    assert seen["prompts"] == []


def test_empty_collection_returns_dont_know_and_no_sources(
    tmp_path, monkeypatch
):
    seen, scorer = patch_all(monkeypatch)
    collection = make_collection(tmp_path, fill=False)
    result = pipeline.answer_with_sources("zebra", collection, None, scorer)
    assert result == {"answer": "I don't know.", "sources": []}
    assert seen["embed"] == [] and seen["pairs"] == []
    assert seen["prompts"] == []
