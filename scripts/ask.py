import argparse
from pathlib import Path

from docqa.loader import extract_pages
from docqa.chunking import chunk_pages
from docqa.embeddings import embed_texts, load_model
from docqa.pipeline import answer


size = 1000
overlap = 200
k = 3

parser = argparse.ArgumentParser()
parser.add_argument("pdf", type=Path)
parser.add_argument("question")
args = parser.parse_args()
question = args.question

pages = extract_pages(args.pdf)
chunks = chunk_pages(pages, args.pdf.name, size, overlap)
model = load_model()
vecs = embed_texts(model, [c.text for c in chunks])
print(answer(question, chunks, vecs, model, k))

