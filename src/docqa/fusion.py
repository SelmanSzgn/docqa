def rrf_fuse(rankings: list[list[str]], top_n: int, c: int = 60) -> list[str]:
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    if len(rankings) == 0 or all(len(r) == 0 for r in rankings):
        return []
    scores = {}
    for _ranking in rankings:
        for rank, chunk_id in enumerate(_ranking, start=1):
            if chunk_id in scores.keys():
                scores[chunk_id] += 1 / (c + rank)
            else:
                scores[chunk_id] = 1 / (c + rank)
    return sorted(scores, key=lambda x: (-scores[x], x))[:top_n]
