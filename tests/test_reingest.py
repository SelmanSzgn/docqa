import inspect

import chromadb
import numpy as np
import pymupdf
import pytest

from docqa import pipeline


def make_pdf(path, pages_text):
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def make_collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    return client.get_or_create_collection("test")


def patch_embed(monkeypatch):
    def fake_embed(model, texts):
        return np.array([[float(i), 1.0] for i in range(len(texts))])

    monkeypatch.setattr(pipeline, "embed_texts", fake_embed)


def stored(collection):
    metadatas = collection.get()["metadatas"]
    return sorted((m["source"], m["page"]) for m in metadatas)


def test_default_size_and_overlap_are_500_and_100():
    params = inspect.signature(pipeline.ingest_pdf).parameters
    assert params["size"].default == 500
    assert params["overlap"].default == 100


def test_reingest_replaces_old_chunks_of_same_source(tmp_path, monkeypatch):
    patch_embed(monkeypatch)
    pdf = tmp_path / "doc.pdf"
    collection = make_collection(tmp_path)
    make_pdf(pdf, ["alpha", "beta"])
    assert pipeline.ingest_pdf(pdf, collection, None) == 2
    make_pdf(pdf, ["gamma"])
    assert pipeline.ingest_pdf(pdf, collection, None) == 1
    assert collection.count() == 1
    assert stored(collection) == [("doc.pdf", 1)]
    assert collection.get()["documents"][0].strip() == "gamma"


def test_reingest_with_different_chunking_leaves_no_stale_chunks(
    tmp_path, monkeypatch
):
    patch_embed(monkeypatch)
    pdf = tmp_path / "doc.pdf"
    collection = make_collection(tmp_path)
    make_pdf(pdf, ["abcdefghij"])
    first = pipeline.ingest_pdf(pdf, collection, None, size=4, overlap=1)
    assert first > 1
    second = pipeline.ingest_pdf(pdf, collection, None, size=100, overlap=0)
    assert second == 1
    assert collection.count() == 1


def test_reingest_keeps_other_sources(tmp_path, monkeypatch):
    patch_embed(monkeypatch)
    doc = tmp_path / "doc.pdf"
    other = tmp_path / "other.pdf"
    collection = make_collection(tmp_path)
    make_pdf(other, ["delta"])
    make_pdf(doc, ["alpha", "beta"])
    pipeline.ingest_pdf(other, collection, None)
    pipeline.ingest_pdf(doc, collection, None)
    make_pdf(doc, ["gamma"])
    pipeline.ingest_pdf(doc, collection, None)
    assert stored(collection) == [("doc.pdf", 1), ("other.pdf", 1)]


def test_reingest_blank_version_removes_old_chunks(tmp_path, monkeypatch):
    patch_embed(monkeypatch)
    pdf = tmp_path / "doc.pdf"
    collection = make_collection(tmp_path)
    make_pdf(pdf, ["alpha"])
    assert pipeline.ingest_pdf(pdf, collection, None) == 1
    make_pdf(pdf, [""])
    assert pipeline.ingest_pdf(pdf, collection, None) == 0
    assert collection.count() == 0


def test_failed_embedding_keeps_old_chunks(tmp_path, monkeypatch):
    patch_embed(monkeypatch)
    pdf = tmp_path / "doc.pdf"
    collection = make_collection(tmp_path)
    make_pdf(pdf, ["alpha", "beta"])
    pipeline.ingest_pdf(pdf, collection, None)

    def broken_embed(model, texts):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "embed_texts", broken_embed)
    make_pdf(pdf, ["gamma"])
    with pytest.raises(RuntimeError, match="boom"):
        pipeline.ingest_pdf(pdf, collection, None)
    assert stored(collection) == [("doc.pdf", 1), ("doc.pdf", 2)]
