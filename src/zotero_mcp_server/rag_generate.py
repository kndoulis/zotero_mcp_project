# ABOUTME: Retrieval-augmented generation over the indexed Zotero library.
# ABOUTME: Retrieves relevant chunks, builds a cite-your-sources prompt, and calls Claude for a synthesized answer.
from typing import Dict, List, Tuple

from .llm_client import DEFAULT_MODEL, call_claude, extract_text
from .semantic_search import PaperChunk, SemanticSearchEngine

SYSTEM_PROMPT = """You are a research assistant answering questions using only the provided excerpts from the user's Zotero library.

Rules:
- Answer only using information from the provided sources. If the sources don't contain enough information to answer, say so plainly instead of guessing.
- After every claim, cite the source it came from using its paper key in square brackets, e.g. [ABCD1234].
- If multiple sources support a claim, cite all of them, e.g. [ABCD1234][WXYZ5678].
- Do not fabricate citations to keys that are not in the provided sources."""


def _format_sources(chunks: List[Tuple[PaperChunk, float]], search_engine: SemanticSearchEngine) -> str:
    """Render retrieved chunks into a source block keyed by paper key, for the prompt."""
    blocks = []
    for chunk, score in chunks:
        paper = search_engine.paper_index.get(chunk.paper_key)
        title = paper.title if paper else "Unknown title"
        blocks.append(
            f"[{chunk.paper_key}] {title} - section: {chunk.section_title} (relevance: {score:.3f})\n{chunk.text}"
        )
    return "\n---\n".join(blocks)


async def ask_library(
    query: str,
    search_engine: SemanticSearchEngine,
    max_chunks: int = 8,
    min_similarity: float = 0.0,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> Dict:
    """Retrieve relevant chunks for a query and synthesize a cited answer with Claude.

    Returns a dict with the generated answer text and the list of sources actually
    retrieved (paper key, title, section, relevance score), so callers can render
    both the answer and a source list without re-deriving it from the citations.
    """
    chunks = await search_engine.search_chunks(query, max_results=max_chunks, min_similarity=min_similarity)

    if not chunks:
        return {
            "answer": "No relevant sources were found in your library for this question.",
            "sources": [],
        }

    sources_block = _format_sources(chunks, search_engine)
    user_message = (
        f"Question: {query}\n\n"
        f"Sources:\n{sources_block}\n\n"
        "Answer the question using only these sources, citing paper keys as instructed."
    )

    response = call_claude(
        messages=[{"role": "user", "content": user_message}],
        system=SYSTEM_PROMPT,
        model=model,
        max_tokens=max_tokens,
        label="ask_library",
    )

    return {
        "answer": extract_text(response),
        "sources": [
            {
                "paper_key": chunk.paper_key,
                "title": (
                    search_engine.paper_index[chunk.paper_key].title
                    if chunk.paper_key in search_engine.paper_index
                    else "Unknown title"
                ),
                "section_title": chunk.section_title,
                "relevance": score,
            }
            for chunk, score in chunks
        ],
    }
