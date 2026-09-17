# ABOUTME: Unit tests for the shared Anthropic API client wrapper and usage logging.
# ABOUTME: Tests client creation, error handling, API calls, and token usage logging.
import json
from unittest.mock import MagicMock, patch

import pytest

from src.zotero_mcp_server import llm_client


@pytest.fixture(autouse=True)
def reset_client():
    """Reset the module-level cached client before and after each test."""
    llm_client._client = None
    yield
    llm_client._client = None


def test_get_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        llm_client.get_client()


def test_get_client_returns_cached_instance(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("src.zotero_mcp_server.llm_client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_anthropic_cls.return_value = MagicMock()
        client1 = llm_client.get_client()
        client2 = llm_client.get_client()
        assert client1 is client2
        mock_anthropic_cls.assert_called_once()


def test_call_claude_logs_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    mock_message = MagicMock()
    mock_message.usage.input_tokens = 42
    mock_message.usage.output_tokens = 7
    mock_message.content = [MagicMock(type="text", text="hello")]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    with patch("src.zotero_mcp_server.llm_client.anthropic.Anthropic", return_value=mock_client):
        response = llm_client.call_claude(
            messages=[{"role": "user", "content": "hi"}],
            system="be nice",
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            label="test_call",
            data_dir=str(tmp_path),
        )

    assert response is mock_message
    mock_client.messages.create.assert_called_once_with(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        system="be nice",
    )

    log_path = tmp_path / "usage_log.jsonl"
    assert log_path.exists()
    entry = json.loads(log_path.read_text().strip().splitlines()[-1])
    assert entry["label"] == "test_call"
    assert entry["model"] == "claude-haiku-4-5-20251001"
    assert entry["input_tokens"] == 42
    assert entry["output_tokens"] == 7


def test_extract_text_concatenates_text_blocks():
    response = MagicMock()
    response.content = [
        MagicMock(type="text", text="Hello "),
        MagicMock(type="tool_use", text=None),
        MagicMock(type="text", text="world"),
    ]
    assert llm_client.extract_text(response) == "Hello world"
