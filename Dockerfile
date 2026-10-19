FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md .
COPY src ./src
RUN uv sync --frozen --no-dev
COPY scripts ./scripts
ENV HOST=0.0.0.0
CMD ["uv", "run", "--no-sync", "python", "scripts/serve.py"]
