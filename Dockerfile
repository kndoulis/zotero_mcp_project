# ABOUTME: Container image for the ask_library RAG pipeline, served via FastAPI/uvicorn.
# ABOUTME: Bakes in a pre-built search index - run `python scripts/reindex.py` locally before building.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY src/ ./src/

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir .

# Bakes in the persisted FAISS index built by scripts/reindex.py. Excludes usage_log.jsonl
# and agent_notes.txt (personal artifacts, not needed at runtime) via .dockerignore.
COPY data/ ./data/

EXPOSE 8080

CMD uvicorn zotero_mcp_server.api:app --host 0.0.0.0 --port ${PORT:-8080}
