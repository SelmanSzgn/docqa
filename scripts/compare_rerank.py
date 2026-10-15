import sys

import chromadb

from docqa.embeddings import load_model
from docqa.evaluation import evaluate, load_dataset
from docqa.pipeline import retrieve_chunks_hybrid
from docqa.reranker import load_scorer, rerank

dataset_path = "eval/attention_questions.json"
dataset = load_dataset(dataset_path)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("docs")

if collection.count() == 0:
    sys.exit("No documents ingested yet. Run scripts/ingest.py first.")

model = load_model()
scorer = load_scorer()

res_hybrid = "hybrid "
res_rerank = "rerank "
for k in (1, 3, 5):
    metrics_hybrid = evaluate(
        lambda q, collection=collection: [
            c.page for c in retrieve_chunks_hybrid(q, collection, model, 5)
        ],
        dataset,
        k,
    )
    res_hybrid += f"hit@{k}={metrics_hybrid['hit_rate']:.3f} "

    metrics_rerank = evaluate(
        lambda q, collection=collection: [
            c.page
            for c in rerank(
                q, retrieve_chunks_hybrid(q, collection, model, 20), scorer, 5
            )
        ],
        dataset,
        k,
    )
    res_rerank += f"hit@{k}={metrics_rerank['hit_rate']:.3f} "

res_hybrid += f"mrr={metrics_hybrid['mrr']:.3f}"
res_rerank += f"mrr={metrics_rerank['mrr']:.3f}"

print(res_hybrid)
print(res_rerank)
