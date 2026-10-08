import numpy as np


def top_k(
    query_vec: np.ndarray, vecs: np.ndarray, k: int
) -> list[tuple[int, float]]:
    if k <= 0:
        raise ValueError(f"Value of k must be positive, k={k} was given.")
    scores = vecs @ query_vec
    order = np.argsort(scores)
    topk_idx = order[::-1][:k]
    topk = [(int(i), scores[i]) for i in topk_idx]
    return topk
