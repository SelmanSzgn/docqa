import numpy as np

from docqa.chunking import Chunk
from docqa.embeddings import embed_texts
from docqa.llm import generate
from docqa.prompt import build_prompt
from docqa.search import top_k
from docqa.store import query_chunks


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

