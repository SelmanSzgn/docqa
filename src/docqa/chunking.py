from dataclasses import dataclass


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if overlap < 0:
        raise ValueError("Overlap value must be non negative.")
    if size <= 0:
        raise ValueError("Chunk size value must be positive.")
    if text == "":
        return []
    step = size - overlap
    if step <= 0:
        raise ValueError("Chunk size value must be greater than overlap.")
    if len(text) == 1:
        return [text]
    chunks = []
    for start in range(0, len(text), step):
        _chunk = text[start : start + size]
        chunks.append(_chunk)
        if start + size >= len(text):
            break
    return chunks


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int


def chunk_pages(
    pages: list[tuple[int, str]], source: str, size: int, overlap: int
) -> list[Chunk]:
    all_chunks = list()
    for page, text in pages:
        _chunks = chunk_text(text, size, overlap)
        for _c in _chunks:
            all_chunks.append(Chunk(text=_c, source=source, page=page))
    return all_chunks
