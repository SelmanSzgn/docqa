import os
import sys

import chromadb
import uvicorn

from docqa.api import create_app
from docqa.embeddings import load_model
from docqa.pipeline import answer_reranked, answer_with_sources
from docqa.reranker import load_scorer

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("docs")
if collection.count() == 0:
    sys.exit("No documents ingested yet. Run scripts/ingest.py first.")
model = load_model()
scorer = load_scorer()
app = create_app(
    lambda q: answer_reranked(q, collection, model, scorer, k=5),
    lambda q: answer_with_sources(q, collection, model, scorer, k=5),
)
host = os.environ.get("HOST", "127.0.0.1")
uvicorn.run(app, host=host, port=8000)
