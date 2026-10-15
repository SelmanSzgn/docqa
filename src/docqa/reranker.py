from sentence_transformers import CrossEncoder

from docqa.chunking import Chunk


def rerank(question: str, chunks: list[Chunk], scorer, k: int) -> list[Chunk]:
    if k <= 0:
        raise ValueError("k must be positive.")
    if len(chunks) == 0:
        return []
    scores = scorer([(question, c.text) for c in chunks])
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [chunks[i] for i in order]


def load_scorer():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").predict
