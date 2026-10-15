import sys
import chromadb
import uvicorn

from docqa.embeddings import load_model
from docqa.reranker import load_scorer
from docqa.api import create_app
from docqa.pipeline import answer_reranked

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("docs")
if collection.count() == 0:
    sys.exit("No document ingested yet. Run scripts/ingest.py first.")
model = load_model()
scorer = load_scorer()
app = create_app(lambda q: answer_reranked(q, collection, model, scorer, k=5))
uvicorn.run(app, host="127.0.0.1", port=8000)

