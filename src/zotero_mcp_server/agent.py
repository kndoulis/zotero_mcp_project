# ABOUTME: Standalone CLI agent that uses a Claude tool-use loop to search, summarize, and take notes on the Zotero library.
# ABOUTME: Runs independently of the MCP server; prints each tool call and result so the loop is visible/debuggable.
import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Tuple

from .llm_client import DEFAULT_MODEL, call_claude, extract_text
from .semantic_search import SemanticSearchEngine

MAX_ITERATIONS = 8
NOTES_PATH = Path("data") / "agent_notes.txt"

SYSTEM_PROMPT = (
    "You are a research assistant agent with access to the user's Zotero library. "
    "Use the available tools to accomplish the user's goal, then give a final answer as plain "
    "text (no further tool calls) once you're done. Be concise and cite paper keys in square "
    "brackets when you reference specific papers."
)

TOOLS = [
    {
        "name": "search_papers",
        "description": "Search the user's Zotero library for papers relevant to a topic, using semantic search.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The research topic or question to search for"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5,
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "summarize_paper",
        "description": "Summarize a specific paper by its Zotero item key.",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_key": {"type": "string", "description": "The Zotero item key of the paper to summarize"},
                "focus": {
                    "type": "string",
                    "description": "Optional: an aspect to focus the summary on, e.g. 'methodology'",
                    "default": "",
                },
            },
            "required": ["paper_key"],
        },
    },
    {
        "name": "save_note",
        "description": "Save a short text note to a local notes file, e.g. a finding or summary worth keeping.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "A short title for the note"},
                "content": {"type": "string", "description": "The note's content"},
            },
            "required": ["title", "content"],
        },
    },
]


async def _search_papers_tool(search_engine: SemanticSearchEngine, arguments: Dict) -> str:
    topic = arguments["topic"]
    max_results = arguments.get("max_results", 5)
    results = await search_engine.search_by_topic(topic, max_results=max_results, min_similarity=0.0)

    if not results:
        return f"No papers found for topic '{topic}'."

    lines = [f"Found {len(results)} papers for '{topic}':"]
    for paper, score in results:
        lines.append(f"- [{paper.key}] {paper.title} (relevance: {score:.3f})")
    return "\n".join(lines)


async def _summarize_paper_tool(search_engine: SemanticSearchEngine, arguments: Dict, model: str) -> str:
    paper_key = arguments["paper_key"]
    focus = arguments.get("focus", "")
    paper = search_engine.paper_index.get(paper_key)
    if not paper:
        return f"No paper found with key '{paper_key}'."

    content_parts = [f"Title: {paper.title}"]
    if paper.abstract:
        content_parts.append(f"Abstract: {paper.abstract}")
    if paper.full_text:
        content_parts.append(f"Full text (excerpt): {paper.full_text[:3000]}")
    paper_text = "\n\n".join(content_parts)

    focus_instruction = f" Focus specifically on: {focus}." if focus else ""
    response = call_claude(
        messages=[{
            "role": "user",
            "content": f"Summarize this paper in 3-5 sentences.{focus_instruction}\n\n{paper_text}",
        }],
        model=model,
        max_tokens=512,
        label="summarize_paper",
    )
    return extract_text(response)


def _save_note_tool(arguments: Dict) -> str:
    title = arguments["title"]
    content = arguments["content"]
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTES_PATH, "a") as f:
        f.write(f"## {title}\n{content}\n\n")
    return f"Note '{title}' saved to {NOTES_PATH}."


async def _execute_tool(name: str, arguments: Dict, search_engine: SemanticSearchEngine, model: str) -> Tuple[str, bool]:
    """Run a tool by name. Returns (result_text, is_error)."""
    try:
        if name == "search_papers":
            return await _search_papers_tool(search_engine, arguments), False
        elif name == "summarize_paper":
            return await _summarize_paper_tool(search_engine, arguments, model), False
        elif name == "save_note":
            return _save_note_tool(arguments), False
        else:
            return f"Unknown tool: {name}", True
    except Exception as e:
        return f"Error executing tool '{name}': {e}", True


async def run_agent(goal: str, model: str = DEFAULT_MODEL, max_iterations: int = MAX_ITERATIONS) -> str:
    """Run the agentic tool-use loop on a goal until Claude gives a final answer or the iteration cap is hit."""
    search_engine = SemanticSearchEngine()
    if not search_engine.load_index():
        raise RuntimeError(
            "No persisted index found. Run `python scripts/reindex.py` first to build the library index."
        )

    messages: List[Dict] = [{"role": "user", "content": goal}]

    for iteration in range(1, max_iterations + 1):
        print(f"\n=== Agent iteration {iteration}/{max_iterations} ===")

        response = call_claude(
            messages=messages,
            system=SYSTEM_PROMPT,
            model=model,
            max_tokens=1024,
            tools=TOOLS,
            label="agent_loop",
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"[Claude] {block.text.strip()}")

        tool_use_blocks = [block for block in response.content if block.type == "tool_use"]

        if response.stop_reason != "tool_use" or not tool_use_blocks:
            final_answer = extract_text(response)
            print(f"\n=== Final answer ===\n{final_answer}")
            return final_answer

        tool_results = []
        for block in tool_use_blocks:
            print(f"[Tool call] {block.name}({json.dumps(block.input)})")
            result_text, is_error = await _execute_tool(block.name, block.input, search_engine, model)
            status = "ERROR" if is_error else "OK"
            preview = result_text if len(result_text) <= 500 else result_text[:500] + "..."
            print(f"[Tool result: {status}] {preview}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
                "is_error": is_error,
            })

        messages.append({"role": "user", "content": tool_results})

    print(f"\n=== Reached max iterations ({max_iterations}) without a final answer ===")
    return f"Agent stopped: reached the maximum number of iterations ({max_iterations}) without producing a final answer."


def main():
    parser = argparse.ArgumentParser(description="Run the Zotero research agent on a goal.")
    parser.add_argument("goal", type=str, help="e.g. 'find papers on GANs and summarize their methodology'")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Claude model to use (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--max-iterations", type=int, default=MAX_ITERATIONS, help=f"Hard cap on loop turns (default: {MAX_ITERATIONS})"
    )
    args = parser.parse_args()

    asyncio.run(run_agent(args.goal, model=args.model, max_iterations=args.max_iterations))


if __name__ == "__main__":
    main()
