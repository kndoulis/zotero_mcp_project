# ABOUTME: FastAPI wrapper exposing the ask_library RAG pipeline as a single POST /ask endpoint.
# ABOUTME: Loads a persisted search index once at startup; run scripts/reindex.py before starting/deploying this.
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .llm_client import DEFAULT_MODEL
from .rag_generate import ask_library
from .semantic_search import SemanticSearchEngine

search_engine: Optional[SemanticSearchEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_engine
    search_engine = SemanticSearchEngine()
    if not search_engine.load_index():
        raise RuntimeError(
            "No persisted index found in data/. Run `python scripts/reindex.py` before starting the API."
        )
    yield


app = FastAPI(title="Zotero RAG API", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str
    max_chunks: int = 8
    model: str = DEFAULT_MODEL


class Source(BaseModel):
    paper_key: str
    title: str
    section_title: str
    relevance: float


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "papers_indexed": len(search_engine.paper_index) if search_engine else 0,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search index not loaded")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    result = await ask_library(
        request.query,
        search_engine,
        max_chunks=request.max_chunks,
        model=request.model,
    )
    return AskResponse(answer=result["answer"], sources=result["sources"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
