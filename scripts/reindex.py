# ABOUTME: Standalone script that runs the Zotero library indexing pipeline directly, without an MCP client.
# ABOUTME: Prints progress and a summary so indexing/chunking can be verified against the real library.
import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import zotero_mcp_server.main as mcp_main


async def run(force_refresh: bool) -> None:
    await mcp_main.initialize_clients(force_refresh=force_refresh)

    search_engine = mcp_main.search_engine
    papers_with_fulltext = sum(1 for p in search_engine.paper_index.values() if p.full_text)
    section_counts = Counter(chunk.section_title for chunk in search_engine.chunks.values())

    print("\n--- Index summary ---")
    print(f"Papers fetched from Zotero: {len(mcp_main.indexed_papers)}")
    print(f"Papers with full_text:      {papers_with_fulltext}")
    print(f"Document-level index size:  {search_engine.faiss_index.ntotal}")
    print(f"Chunk-level index size:     {search_engine.chunk_faiss_index.ntotal}")
    print(f"Section title distribution: {section_counts.most_common(10)}")
    print(f"Index persisted to:         {search_engine.data_dir.resolve()}")

    await mcp_main.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Zotero indexing pipeline standalone and print a summary.")
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Load a persisted index if one exists instead of forcing a full re-fetch/re-index.",
    )
    args = parser.parse_args()

    asyncio.run(run(force_refresh=not args.no_refresh))
