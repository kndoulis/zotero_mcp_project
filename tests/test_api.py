# ABOUTME: Unit tests for the FastAPI wrapper around the ask_library RAG pipeline.
# ABOUTME: Mocks the search engine and RAG call so these tests make no real API calls or model loads.
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """A TestClient with SemanticSearchEngine patched out, so app startup needs no real model/index."""
    with patch("src.zotero_mcp_server.api.SemanticSearchEngine") as mock_engine_cls:
        mock_engine = MagicMock()
        mock_engine.load_index.return_value = True
        mock_engine.paper_index = {"PAPER1": MagicMock()}
        mock_engine_cls.return_value = mock_engine

        from src.zotero_mcp_server.api import app

        with TestClient(app) as test_client:
            yield test_client, mock_engine


def test_health_endpoint_reports_indexed_paper_count(client):
    test_client, engine = client

    response = test_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["papers_indexed"] == 1


def test_ask_endpoint_returns_answer_and_sources(client):
    test_client, engine = client
    mock_result = {
        "answer": "ML improves diagnosis [PAPER1].",
        "sources": [{
            "paper_key": "PAPER1",
            "title": "Machine Learning in Healthcare",
            "section_title": "Abstract",
            "relevance": 0.9,
        }],
    }

    with patch("src.zotero_mcp_server.api.ask_library", new=AsyncMock(return_value=mock_result)) as mock_ask:
        response = test_client.post("/ask", json={"query": "does ML help diagnosis?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "ML improves diagnosis [PAPER1]."
    assert body["sources"][0]["paper_key"] == "PAPER1"
    mock_ask.assert_awaited_once()
    call_kwargs = mock_ask.call_args.kwargs
    assert call_kwargs["max_chunks"] == 8  # default


def test_ask_endpoint_passes_through_max_chunks_and_model(client):
    test_client, engine = client
    mock_result = {"answer": "No sources found.", "sources": []}

    with patch("src.zotero_mcp_server.api.ask_library", new=AsyncMock(return_value=mock_result)) as mock_ask:
        response = test_client.post(
            "/ask", json={"query": "obscure topic", "max_chunks": 3, "model": "claude-sonnet-5"}
        )

    assert response.status_code == 200
    call_kwargs = mock_ask.call_args.kwargs
    assert call_kwargs["max_chunks"] == 3
    assert call_kwargs["model"] == "claude-sonnet-5"


def test_ask_endpoint_rejects_empty_query(client):
    test_client, engine = client

    response = test_client.post("/ask", json={"query": "   "})

    assert response.status_code == 400


def test_app_startup_fails_without_persisted_index():
    with patch("src.zotero_mcp_server.api.SemanticSearchEngine") as mock_engine_cls:
        mock_engine = MagicMock()
        mock_engine.load_index.return_value = False
        mock_engine_cls.return_value = mock_engine

        from src.zotero_mcp_server.api import app

        with pytest.raises(RuntimeError, match="reindex"):
            with TestClient(app):
                pass
