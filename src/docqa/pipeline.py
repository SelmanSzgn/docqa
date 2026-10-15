from pathlib import Path

import numpy as np

from docqa.chunking import Chunk, chunk_pages
from docqa.embeddings import embed_texts
from docqa.fusion import rrf_fuse
from docqa.lexical import bm25_top_k
from docqa.llm import generate
from docqa.loader import extract_pages
from docqa.prompt import build_prompt
from docqa.reranker import rerank
from docqa.search import top_k
from docqa.store import add_chunks, query_chunks


def answer(
    question: str, chunks: list[Chunk], vecs: np.ndarray, model, k: int = 3
) -> str:
    q = embed_texts(model, [question])[0]
    topk = top_k(q, vecs, k)
    best_chunks = [chunks[i] for i, _ in topk]
    prompt = build_prompt(question, best_chunks)
    return generate(prompt)


def answer_from_store(question: str, collection, model, k: int = 3) -> str:
    if k <= 0:
        raise ValueError(f"k must be positive; k={k} was given.")
    q = embed_texts(model, [question])[0]
    chunks = query_chunks(collection, q, k)
    prompt = build_prompt(question, chunks)
    return generate(prompt)


def ingest_pdf(
    path: Path, collection, model, size: int = 500, overlap: int = 100
) -> int:
    pages = extract_pages(path)
    chunks = chunk_pages(pages, path.name, size, overlap)
    n_chunks = len(chunks)
    if n_chunks == 0:
        collection.delete(where={"source": path.name})
        return 0
    vecs = embed_texts(model, [c.text for c in chunks])
    collection.delete(where={"source": path.name})
    add_chunks(collection, chunks, vecs)
    return n_chunks


def retrieve_pages(question: str, collection, model, k: int) -> list[int]:
    if k <= 0:
        raise ValueError(f"k must be positive; k={k} was given.")
    q = embed_texts(model, [question])[0]
    chunks = query_chunks(collection, q, k)
    retrieved = [c.page for c in chunks]
    return retrieved


def retrieve_chunks_hybrid(
    question: str, collection, model, k: int, pool: int = 20
) -> list[Chunk]:
    if k <= 0:
        raise ValueError(f"k must be positive; k={k} was given.")
    if collection.count() == 0:
        return []
    data = collection.get()
    n_chunks = collection.count()
    chunks = []
    for i in range(n_chunks):
        text = data["documents"][i]
        source = data["metadatas"][i]["source"]
        page = data["metadatas"][i]["page"]
        chunks.append(Chunk(text, source, page))
    q = embed_texts(model, [question])[0]
    ids_vec = collection.query(query_embeddings=[q], n_results=pool)["ids"][0]
    texts = [c.text for c in chunks]
    _indices = bm25_top_k(question, texts, pool)
    ids_bm25 = [data["ids"][j] for j in _indices]
    scores = rrf_fuse([ids_vec, ids_bm25], k)
    best_chunks = []
    for chunk_id in scores:
        pos = 0
        for i in range(n_chunks):
            if data["ids"][i] == chunk_id:
                break
            pos += 1
        best_chunks.append(chunks[pos])
    return best_chunks


def answer_hybrid(question: str, collection, model, k: int = 3) -> str:
    if k <= 0:
        raise ValueError(f"k must be positive; k={k} was given.")
    chunks = retrieve_chunks_hybrid(question, collection, model, k)
    if len(chunks) == 0:
        return "I don't know."
    prompt = build_prompt(question, chunks)
    return generate(prompt)


def answer_reranked(
    question: str, collection, model, scorer, k: int = 3, pool: int = 20
) -> str:
    if k <= 0:
        raise ValueError("k must be positive.")
    if pool < k:
        raise ValueError("pool must be greater or equal to k.")
    candidats = retrieve_chunks_hybrid(question, collection, model, pool)
    if len(candidats) == 0:
        return "I don't know."
    chunks = rerank(question, candidats, scorer, k)
    prompt = build_prompt(question, chunks)
    return generate(prompt)
