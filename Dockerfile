FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY scripts ./scripts
COPY README.md .
RUN uv sync --frozen --no-dev
ENV HOST=0.0.0.0
EXPOSE 8000
CMD ["uv", "run", "--no-dev", "python", "scripts/serve.py"]

