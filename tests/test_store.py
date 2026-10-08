import chromadb
import numpy as np

from docqa.chunking import Chunk
from docqa.store import add_chunks, query_chunks

CHUNKS = [
    Chunk("alpha", "doc.pdf", 1),
    Chunk("beta", "doc.pdf", 2),
    Chunk("gamma", "other.pdf", 3),
]
VECS = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])


def make_collection(path):
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection("test")


def test_query_returns_closest_chunks(tmp_path):
    collection = make_collection(tmp_path)
    add_chunks(collection, CHUNKS, VECS)
    result = query_chunks(collection, np.array([1.0, 0.0]), k=2)
    assert result == [CHUNKS[0], CHUNKS[2]]


def test_data_persists_on_disk(tmp_path):
    add_chunks(make_collection(tmp_path), CHUNKS, VECS)
    assert make_collection(tmp_path).count() == 3


def test_adding_twice_does_not_duplicate(tmp_path):
    collection = make_collection(tmp_path)
    add_chunks(collection, CHUNKS, VECS)
    add_chunks(collection, CHUNKS, VECS)
    assert collection.count() == 3


def test_query_with_k_larger_than_collection(tmp_path):
    collection = make_collection(tmp_path)
    add_chunks(collection, CHUNKS, VECS)
    result = query_chunks(collection, np.array([1.0, 0.0]), k=10)
    assert len(result) == 3
