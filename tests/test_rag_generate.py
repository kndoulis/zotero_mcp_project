# ABOUTME: Unit tests for the RAG generation pipeline (retrieval + Claude synthesis with citations).
# ABOUTME: Tests the no-results path and end-to-end ask_library with a mocked Claude call.
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.zotero_mcp_server.rag_generate import ask_library
from src.zotero_mcp_server.semantic_search import PaperChunk
from src.zotero_mcp_server.zotero_client import ZoteroItem


def make_search_engine(chunks_with_scores, papers):
    engine = MagicMock()
    engine.search_chunks = AsyncMock(return_value=chunks_with_scores)
    engine.paper_index = {p.key: p for p in papers}
    return engine


@pytest.mark.asyncio
async def test_ask_library_returns_no_sources_message_when_nothing_retrieved():
    engine = make_search_engine([], [])

    result = await ask_library("what is X?", engine)

    assert result["sources"] == []
    assert "No relevant sources" in result["answer"]


@pytest.mark.asyncio
async def test_ask_library_builds_prompt_and_returns_sources():
    paper = ZoteroItem(
        key="PAPER1",
        title="Machine Learning in Healthcare",
        creators=[],
        item_type="journalArticle",
    )
    chunk = PaperChunk(
        chunk_id="PAPER1::0",
        paper_key="PAPER1",
        chunk_index=0,
        section_title="Abstract",
        text="Machine learning improves diagnosis accuracy.",
    )
    engine = make_search_engine([(chunk, 0.87)], [paper])

    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="ML improves diagnosis [PAPER1].")]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5

    with patch("src.zotero_mcp_server.rag_generate.call_claude", return_value=mock_response) as mock_call:
        result = await ask_library(
            "does ML help diagnosis?", engine, max_chunks=5, model="claude-haiku-4-5-20251001"
        )

    assert result["answer"] == "ML improves diagnosis [PAPER1]."
    assert result["sources"] == [{
        "paper_key": "PAPER1",
        "title": "Machine Learning in Healthcare",
        "section_title": "Abstract",
        "relevance": 0.87,
    }]

    engine.search_chunks.assert_awaited_once_with("does ML help diagnosis?", max_results=5, min_similarity=0.0)

    call_kwargs = mock_call.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
    assert "PAPER1" in call_kwargs["messages"][0]["content"]
    assert "cite" in call_kwargs["system"].lower()
