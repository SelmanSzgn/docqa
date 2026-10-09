import chromadb
import numpy as np
import pymupdf

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
    calls = []

    def fake_embed(model, texts):
        calls.append(texts)
        return np.array([[float(i), 1.0] for i in range(len(texts))])

    monkeypatch.setattr(pipeline, "embed_texts", fake_embed)
    return calls


def pages_stored(collection):
    metadatas = collection.get()["metadatas"]
    return sorted((m["source"], m["page"]) for m in metadatas)


def test_ingest_stores_chunks_with_source_and_pages(tmp_path, monkeypatch):
    patch_embed(monkeypatch)
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf, ["alpha", "beta"])
    collection = make_collection(tmp_path)
    assert pipeline.ingest_pdf(pdf, collection, None) == 2
    assert collection.count() == 2
    assert pages_stored(collection) == [("doc.pdf", 1), ("doc.pdf", 2)]


def test_ingest_twice_does_not_duplicate(tmp_path, monkeypatch):
    patch_embed(monkeypatch)
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf, ["alpha", "beta"])
    collection = make_collection(tmp_path)
    assert pipeline.ingest_pdf(pdf, collection, None) == 2
    assert pipeline.ingest_pdf(pdf, collection, None) == 2
    assert collection.count() == 2


def test_ingest_blank_pdf_returns_zero(tmp_path, monkeypatch):
    calls = patch_embed(monkeypatch)
    pdf = tmp_path / "blank.pdf"
    make_pdf(pdf, [""])
    collection = make_collection(tmp_path)
    assert pipeline.ingest_pdf(pdf, collection, None) == 0
    assert collection.count() == 0
    assert calls == []


def test_ingest_skips_blank_page_in_the_middle(tmp_path, monkeypatch):
    calls = patch_embed(monkeypatch)
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf, ["alpha", "", "gamma"])
    collection = make_collection(tmp_path)
    assert pipeline.ingest_pdf(pdf, collection, None) == 2
    assert pages_stored(collection) == [("doc.pdf", 1), ("doc.pdf", 3)]
    assert len(calls) == 1


def test_ingest_passes_size_and_overlap(tmp_path, monkeypatch):
    patch_embed(monkeypatch)
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf, ["abcdefghij"])
    collection = make_collection(tmp_path)
    n = pipeline.ingest_pdf(pdf, collection, None, size=4, overlap=1)
    assert n > 1
    assert n == collection.count()
    assert all(len(d) <= 4 for d in collection.get()["documents"])
