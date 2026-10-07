from pathlib import Path

import pymupdf


def extract_pages(path: Path) -> list[tuple[int, str]]:
    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            pages.append((i + 1, text))
    return pages
