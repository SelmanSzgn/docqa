import numpy as np

from docqa.chunking import Chunk
from docqa.embeddings import embed_texts
from docqa.llm import generate
from docqa.prompt import build_prompt
from docqa.search import top_k


def answer(
    question: str, chunks: list[Chunk], vecs: np.ndarray, model, k: int = 3
) -> str:
    q = embed_texts(model, [question])[0]
    topk = top_k(q, vecs, k)
    best_chunks = [chunks[i] for i, _ in topk]
    prompt = build_prompt(question, best_chunks)
    return generate(prompt)
