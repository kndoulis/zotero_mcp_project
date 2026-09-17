# ABOUTME: Unit tests for the standalone agentic CLI tool-use loop.
# ABOUTME: Tests each tool handler in isolation and the loop's tool-use/final-answer/max-iteration paths.
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.zotero_mcp_server.agent import (
    _execute_tool,
    _save_note_tool,
    _search_papers_tool,
    _summarize_paper_tool,
    run_agent,
)
from src.zotero_mcp_server.zotero_client import ZoteroItem


@pytest.mark.asyncio
async def test_search_papers_tool_formats_results():
    paper = ZoteroItem(key="P1", title="Test Paper", creators=[], item_type="journalArticle")
    engine = MagicMock()
    engine.search_by_topic = AsyncMock(return_value=[(paper, 0.9)])

    result = await _search_papers_tool(engine, {"topic": "GANs", "max_results": 5})

    assert "P1" in result
    assert "Test Paper" in result
    engine.search_by_topic.assert_awaited_once_with("GANs", max_results=5, min_similarity=0.0)


@pytest.mark.asyncio
async def test_search_papers_tool_handles_no_results():
    engine = MagicMock()
    engine.search_by_topic = AsyncMock(return_value=[])

    result = await _search_papers_tool(engine, {"topic": "nonexistent topic"})

    assert "No papers found" in result


@pytest.mark.asyncio
async def test_summarize_paper_tool_calls_claude_and_returns_summary():
    paper = ZoteroItem(
        key="P1", title="Test Paper", creators=[], item_type="journalArticle", abstract="An abstract."
    )
    engine = MagicMock()
    engine.paper_index = {"P1": paper}

    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="This paper is about X.")]

    with patch("src.zotero_mcp_server.agent.call_claude", return_value=mock_response) as mock_call:
        result = await _summarize_paper_tool(engine, {"paper_key": "P1"}, model="claude-haiku-4-5-20251001")

    assert result == "This paper is about X."
    call_kwargs = mock_call.call_args.kwargs
    assert "Test Paper" in call_kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_summarize_paper_tool_handles_missing_paper():
    engine = MagicMock()
    engine.paper_index = {}

    result = await _summarize_paper_tool(engine, {"paper_key": "MISSING"}, model="claude-haiku-4-5-20251001")

    assert "No paper found" in result


def test_save_note_tool_appends_to_notes_file(tmp_path, monkeypatch):
    notes_path = tmp_path / "notes.txt"
    monkeypatch.setattr("src.zotero_mcp_server.agent.NOTES_PATH", notes_path)

    result = _save_note_tool({"title": "Finding", "content": "GANs are useful for augmentation."})

    assert "saved" in result.lower()
    assert notes_path.exists()
    text = notes_path.read_text()
    assert "Finding" in text
    assert "GANs are useful" in text


@pytest.mark.asyncio
async def test_execute_tool_unknown_name_is_flagged_as_error():
    engine = MagicMock()

    result_text, is_error = await _execute_tool("not_a_real_tool", {}, engine, "claude-haiku-4-5-20251001")

    assert is_error is True
    assert "Unknown tool" in result_text


@pytest.mark.asyncio
async def test_execute_tool_exception_is_flagged_as_error():
    engine = MagicMock()
    engine.search_by_topic = AsyncMock(side_effect=RuntimeError("boom"))

    result_text, is_error = await _execute_tool(
        "search_papers", {"topic": "x"}, engine, "claude-haiku-4-5-20251001"
    )

    assert is_error is True
    assert "boom" in result_text


def _tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "tool_1"):
    block = MagicMock(type="tool_use", input=tool_input, id=tool_id)
    block.name = tool_name  # "name" is a reserved MagicMock constructor kwarg, so set it post-construction
    return MagicMock(stop_reason="tool_use", content=[block])


def _final_response(text: str):
    block = MagicMock(type="text", text=text)
    return MagicMock(stop_reason="end_turn", content=[block])


@pytest.mark.asyncio
async def test_run_agent_executes_tool_then_returns_final_answer():
    with patch("src.zotero_mcp_server.agent.SemanticSearchEngine") as mock_engine_cls:
        mock_engine = MagicMock()
        mock_engine.load_index.return_value = True
        mock_engine.search_by_topic = AsyncMock(return_value=[])
        mock_engine_cls.return_value = mock_engine

        responses = [
            _tool_use_response("search_papers", {"topic": "GANs"}),
            _final_response("Here is my final answer."),
        ]

        with patch("src.zotero_mcp_server.agent.call_claude", side_effect=responses) as mock_call:
            answer = await run_agent("find papers on GANs", max_iterations=8)

    assert answer == "Here is my final answer."
    assert mock_call.call_count == 2
    mock_engine.search_by_topic.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_agent_stops_at_max_iterations():
    with patch("src.zotero_mcp_server.agent.SemanticSearchEngine") as mock_engine_cls:
        mock_engine = MagicMock()
        mock_engine.load_index.return_value = True
        mock_engine.search_by_topic = AsyncMock(return_value=[])
        mock_engine_cls.return_value = mock_engine

        with patch(
            "src.zotero_mcp_server.agent.call_claude",
            side_effect=lambda **kwargs: _tool_use_response("search_papers", {"topic": "x"}),
        ) as mock_call:
            answer = await run_agent("goal that never finishes", max_iterations=3)

    assert mock_call.call_count == 3
    assert "maximum" in answer.lower()


@pytest.mark.asyncio
async def test_run_agent_raises_when_no_persisted_index():
    with patch("src.zotero_mcp_server.agent.SemanticSearchEngine") as mock_engine_cls:
        mock_engine = MagicMock()
        mock_engine.load_index.return_value = False
        mock_engine_cls.return_value = mock_engine

        with pytest.raises(RuntimeError, match="reindex"):
            await run_agent("goal")
