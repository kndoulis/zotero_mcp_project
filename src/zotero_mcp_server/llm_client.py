# ABOUTME: Shared Anthropic API client wrapper with configurable model selection and usage logging.
# ABOUTME: Used by rag_generate.py and agent.py to call Claude with consistent cost tracking.
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Default model for frequent/dev calls. Pass model="claude-sonnet-5" to any of these
# functions to compare quality against the cheaper default.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 2048

_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    """Return a shared Anthropic client, created lazily from ANTHROPIC_API_KEY."""
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set in the environment. "
                "Copy .env.example to .env and fill in your Anthropic API key."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def log_usage(label: str, model: str, usage, data_dir: str = "data") -> None:
    """Append one line of token-usage info to data/usage_log.jsonl for cost tracking."""
    log_dir = Path(data_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": label,
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    with open(log_dir / "usage_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def call_claude(
    messages: List[dict],
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    tools: Optional[List[dict]] = None,
    label: str = "unlabeled",
    data_dir: str = "data",
) -> anthropic.types.Message:
    """Call the Claude Messages API and log input/output token usage for cost tracking."""
    client = get_client()
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools

    response = client.messages.create(**kwargs)
    log_usage(label, model, response.usage, data_dir=data_dir)
    return response


def extract_text(response: anthropic.types.Message) -> str:
    """Concatenate all text blocks in a response into a single string."""
    return "".join(block.text for block in response.content if block.type == "text")
