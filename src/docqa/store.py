import numpy as np

from docqa.chunking import Chunk


def add_chunks(collection, chunks: list[Chunk], vecs: np.ndarray) -> None:
    ids = []
    embeddings = vecs.tolist()
    documents = []
    metadatas = []
    for i, c in enumerate(chunks):
        documents.append(c.text)
        metadatas.append({"source": c.source, "page": c.page})
        ids.append(f"{c.source}:{c.page}:{i}")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query_chunks(collection, query_vec: np.ndarray, k: int) -> list[Chunk]:
    res = collection.query(query_embeddings=[query_vec], n_results=k)
    chunks = []
    for i in range(len(res["documents"][0])):
        text = res["documents"][0][i]
        source = res["metadatas"][0][i]["source"]
        page = int(res["metadatas"][0][i]["page"])
        chunks.append(Chunk(text=text, source=source, page=page))
    return chunks
