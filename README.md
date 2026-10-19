# docqa

![CI](https://github.com/SelmanSzgn/docqa/actions/workflows/ci.yaml/badge.svg)

Question answering over PDF documents (RAG): hybrid search (vectors + BM25),
cross-encoder reranking, local LLM through Ollama, FastAPI service with sources.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) with a model: `ollama pull qwen2.5:3b`
- Docker (optional)

No GPU needed: PyTorch is installed in its CPU version.

## Quick start

Try the project on the "Attention Is All You Need" paper.

```bash
git clone https://github.com/<your-github-user>/docqa.git
cd docqa
uv sync
mkdir -p data
curl -L -o data/attention.pdf https://arxiv.org/pdf/1706.03762

uv run python scripts/ingest.py data/attention.pdf
uv run python scripts/ask.py "What is multi-head attention?"
```

Two small models (embeddings, reranker) are downloaded on first use.
Ollama must be running (`ollama serve` if it is not already a service).

## Evaluation

32 questions in `eval/attention_questions.json`, on `attention.pdf`, each
annotated with the relevant pages. Metrics measure **retrieval only** (is a
relevant page among the first k chunks?): hit@k, and MRR (mean of 1/rank of the
first relevant page, over the 5 retrieved chunks). One question is worth 3.1
points.

### Chunk size (vector search, `all-MiniLM-L6-v2`)

| size / overlap | chunks | hit@1 | hit@3 | hit@5 | MRR   |
|----------------|--------|-------|-------|-------|-------|
| 250 / 50       | 201    | 0.812 | 0.938 | 1.000 | 0.884 |
| **500 / 100**  | 102    | 0.844 | 0.938 | 0.969 | 0.892 |
| 1000 / 200     | 52     | 0.781 | 0.906 | 0.906 | 0.844 |
| 2000 / 400     | 27     | 0.719 | 0.844 | 0.938 | 0.805 |

500 / 100 is the default.

### Retrieval method (chunks 500 / 100)

| method                           | hit@1 | hit@3 | hit@5 | MRR   |
|----------------------------------|-------|-------|-------|-------|
| vector                           | 0.844 | 0.938 | 0.969 | 0.892 |
| hybrid (vector + BM25, RRF)      | 0.875 | 0.969 | 0.969 | 0.911 |
| hybrid + cross-encoder reranking | 0.938 | 0.969 | 1.000 | 0.961 |

The final pipeline retrieves 20 candidates with hybrid search, reranks them,
and keeps the best 5.

### Reproduce

These scripts are built for `attention.pdf` and its questions. The last three
read `chroma_db/`, which must contain only `attention.pdf` ingested with the
default 500 / 100 chunking (use a clean `chroma_db/`, not one containing
your own documents).

```bash
uv run python scripts/compare_chunking.py   # chunk sizes (in-memory, no ingestion needed)
uv run python scripts/evaluate.py           # vector search only
uv run python scripts/compare_retrieval.py  # vector vs hybrid
uv run python scripts/compare_rerank.py     # hybrid vs reranked
```

## Use your own PDF

1. Put your PDF in `data/` and ingest it:

```bash
   mkdir -p data
   cp /path/to/my_document.pdf data/
   uv run python scripts/ingest.py data/my_document.pdf
```

   The script prints `Ingested N chunks from my_document.pdf`.
   Chunks are stored in `chroma_db/`.

2. Start the API (terminal 1), either:

```bash
   uv run python scripts/serve.py
```

   or with Docker (Ollama must run on the host):

```bash
   docker compose up --build
```

   Wait for `Application startup complete`, then check
   `curl http://127.0.0.1:8000/health`.

3. Start the web UI (terminal 2):

```bash
   uv run --group ui streamlit run scripts/ui.py
```

   Open http://localhost:8501, type a question and click **Ask**.
   The answer is displayed with its sources (file and page).

Notes:

- Re-ingesting the same file replaces its previous chunks (no duplicates).
  Renaming a file and ingesting it again creates a second source.
- Several PDFs can be ingested: questions are asked over all of them.
- Scanned PDFs (images without text) produce 0 chunks: OCR is not supported.
- The embedding model (`all-MiniLM-L6-v2`) and the reranker (`ms-marco-MiniLM`)
  are trained mostly on English. Results will be weaker on other languages.
- The local 3B model sometimes gives vague answers and does not always cite
  pages in its text: the API returns the sources separately.

## API

Routes:

| Route               | Body                       | Response                          |
|---------------------|----------------------------|-----------------------------------|
| `GET /health`       |                            | `{"status": "ok"}`                |
| `POST /ask`         | `{"question": "..."}`      | `{"answer": "..."}`               |
| `POST /ask/sources` | `{"question": "..."}`      | `{"answer": "...", "sources": [...]}` |

A question must be a non-blank string of at most 2000 characters (otherwise 422).
If Ollama is unreachable, the API answers 502.

```bash
curl -X POST http://127.0.0.1:8000/ask/sources \
  -H "Content-Type: application/json" \
  -d '{"question": "What is multi-head attention?"}'
```

Interactive docs: http://127.0.0.1:8000/docs

## Docker

`docker compose up --build` builds the image and starts the API on port 8000.

- Ingest first: the vector store `./chroma_db` is mounted in the container.
- The container reaches Ollama on the host through `host.docker.internal`.
  If the connection is refused, restart Ollama with
  `OLLAMA_HOST=0.0.0.0 ollama serve`.
- The two models are downloaded on the first start of the container.

## Tests

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Configuration

| Variable     | Default                               | Role               |
|--------------|---------------------------------------|--------------------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama endpoint    |
| `HOST`       | `127.0.0.1`                           | API listen address |
