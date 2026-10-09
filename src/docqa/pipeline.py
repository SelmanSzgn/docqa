from pathlib import Path

import numpy as np

from docqa.chunking import Chunk, chunk_pages
from docqa.embeddings import embed_texts
from docqa.llm import generate
from docqa.loader import extract_pages
from docqa.prompt import build_prompt
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
    path: Path, collection, model, size: int = 1000, overlap: int = 200
) -> int:
    pages = extract_pages(path)
    chunks = chunk_pages(pages, path.name, size, overlap)
    n_chunks = len(chunks)
    if n_chunks == 0:
        return 0
    vecs = embed_texts(model, [c.text for c in chunks])
    add_chunks(collection, chunks, vecs)
    return n_chunks
