from pathlib import Path

import chromadb

from docqa.embeddings import load_model
from docqa.evaluation import evaluate, load_dataset
from docqa.pipeline import ingest_pdf, retrieve_pages

dataset_path = "eval/attention_questions.json"
dataset = load_dataset(dataset_path)

model = load_model()

client = chromadb.EphemeralClient()

for size, overlap in [(250, 50), (500, 100), (1000, 200), (2000, 400)]:
    collection = client.get_or_create_collection(f"s{size}_o{overlap}")
    ingest_pdf(
        Path("data/attention.pdf"),
        collection,
        model,
        size=size,
        overlap=overlap,
    )
    n_chunks = collection.count()
    res = f"size={size} overlap={overlap} chunks={n_chunks} "
    for k in [1, 3, 5]:
        metrics = evaluate(
            lambda q, collection=collection: retrieve_pages(
                q, collection, model, 5
            ),
            dataset,
            k,
        )
        res += f"hit@{k}={metrics['hit_rate']:.3f} "
    res += f"mrr={metrics['mrr']:.3f}"
    print(res)
