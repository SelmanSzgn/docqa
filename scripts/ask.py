import argparse
import sys

import chromadb

from docqa.embeddings import load_model
from docqa.pipeline import answer_from_store

size = 1000
overlap = 200
k = 3

parser = argparse.ArgumentParser()
parser.add_argument("question")
args = parser.parse_args()
question = args.question

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("docs")

if collection.count() == 0:
    sys.exit("No documents ingested yet. Run scripts/ingest.py first.")

model = load_model()

print(answer_from_store(question, collection, model, k))
