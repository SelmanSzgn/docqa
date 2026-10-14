
import chromadb

from docqa.embeddings import load_model
from docqa.evaluation import evaluate, load_dataset
from docqa.pipeline import retrieve_chunks_hybrid, retrieve_pages

dataset_path = "eval/attention_questions.json"
dataset = load_dataset(dataset_path)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("docs")

model = load_model()

res_vec = "vector "
res_hybrid = "hybrid "
for k in (1, 3, 5):
    metrics_vec = evaluate(
        lambda q, collection=collection: retrieve_pages(
            q, collection, model, 5
        ),
        dataset,
        k,
    )
    metrics_hybrid = evaluate(
        lambda q, collection=collection: [
            c.page for c in retrieve_chunks_hybrid(q, collection, model, 5)
        ],
        dataset,
        k,
    )
    res_vec += f"hit@{k}={metrics_vec['hit_rate']:.3f} "
    res_hybrid += f"hit@{k}={metrics_hybrid['hit_rate']:.3f} "
res_vec += f"mrr={metrics_vec['mrr']:.3f}"
res_hybrid += f"mrr={metrics_hybrid['mrr']:.3f}"
print(res_vec)
print(res_hybrid)
