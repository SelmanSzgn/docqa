import re
from rank_bm25 import BM25Okapi
import numpy as np


def tokenize(text: str) -> list[str]:
    if not any(c.isalnum() for c in text):
        return []
    tokens = re.findall(r"\w+", text.lower())
    return tokens


def bm25_top_k(query: str, texts: list[str], k: int) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive.")

    if len(texts) == 0:
        return []

    texts_tok = [tokenize(t) for t in texts]
    query_tok = tokenize(query)
    
    if len(query_tok) == 0:
        return []
    if all([len(t) == 0 for t in texts_tok]):
        return []

    bm25 = BM25Okapi(texts_tok)
    scores = bm25.get_scores(query_tok)
    order = np.argsort(scores)[::-1][:k]
    topk = list()
    for j in order:
        if scores[j] > 0:
            topk.append(int(j))
    return topk
    

