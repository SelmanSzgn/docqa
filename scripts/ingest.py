import argparse
from pathlib import Path

import chromadb

from docqa.embeddings import load_model
from docqa.pipeline import ingest_pdf

parser = argparse.ArgumentParser()
parser.add_argument("pdf", type=Path)
args = parser.parse_args()
path = args.pdf

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("docs")

model = load_model()

n_chunks = ingest_pdf(args.pdf, collection, model)

print(f"Ingested {n_chunks} chunks from {args.pdf.name}")
