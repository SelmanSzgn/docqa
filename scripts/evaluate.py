import sys

import chromadb

from docqa.embeddings import load_model
from docqa.evaluation import evaluate, load_dataset
from docqa.pipeline import retrieve_pages

dataset_path = "eval/attention_questions.json"
dataset = load_dataset(dataset_path)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("docs")
if collection.count() == 0:
    sys.exit("No documents ingested yet. Run scripts/ingest.py first.")
model = load_model()
for k in [1, 3, 5]:
    metrics = evaluate(
        lambda q: retrieve_pages(q, collection, model, 5), dataset, k
    )
    print(f"k={k} hit_rate={metrics['hit_rate']:.3f} mrr={metrics['mrr']:.3f}")
