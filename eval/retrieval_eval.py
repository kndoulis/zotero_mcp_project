# ABOUTME: Retrieval evaluation for the Phase 1 document-level FAISS search (search_by_topic).
# ABOUTME: Computes precision@3 and precision@5 against a hand-written query -> relevant-paper(s) test set.
import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

from zotero_mcp_server.semantic_search import SemanticSearchEngine

QUERY_SET_PATH = Path(__file__).parent / "labeled_data" / "retrieval_queries.json"
RESULTS_PATH = Path(__file__).parent / "results" / "retrieval_eval.md"

MAX_K = 5


async def evaluate(search_engine: SemanticSearchEngine, queries: List[Dict]) -> List[Dict]:
    rows = []
    for item in queries:
        query = item["query"]
        relevant = set(item["relevant_paper_keys"])

        results = await search_engine.search_by_topic(query, max_results=MAX_K, min_similarity=0.0)
        retrieved_at_5 = [paper.key for paper, _ in results]
        retrieved_at_3 = retrieved_at_5[:3]

        hits_at_3 = len(set(retrieved_at_3) & relevant)
        hits_at_5 = len(set(retrieved_at_5) & relevant)
        hit_at_1 = 1.0 if retrieved_at_5[:1] and retrieved_at_5[0] in relevant else 0.0

        rows.append({
            "query": query,
            "relevant_paper_keys": sorted(relevant),
            "retrieved_at_5": retrieved_at_5,
            "precision_at_3": hits_at_3 / 3,
            "precision_at_5": hits_at_5 / 5,
            "hit_at_1": hit_at_1,
        })
    return rows


def write_markdown_report(rows: List[Dict], out_path: Path) -> None:
    n = len(rows)
    mean_p3 = sum(r["precision_at_3"] for r in rows) / n
    mean_p5 = sum(r["precision_at_5"] for r in rows) / n
    mean_hit1 = sum(r["hit_at_1"] for r in rows) / n
    single_answer_rows = [r for r in rows if len(r["relevant_paper_keys"]) == 1]

    lines = ["# Retrieval Evaluation (Document-Level FAISS Search)\n"]
    lines.append(
        f"{n} hand-written queries evaluated against `search_by_topic` "
        "(the Phase 1 document-level FAISS index).\n"
    )
    lines.append(
        f"**Mean precision@3: {mean_p3:.1%}**  \n**Mean precision@5: {mean_p5:.1%}**  \n"
        f"**Mean hit@1 (gold paper ranked first): {mean_hit1:.1%}**\n"
    )
    lines.append(
        "Precision@k is mechanically bounded by `|relevant papers| / k`: for the "
        f"{len(single_answer_rows)} queries in this set with only one relevant paper, even a perfect "
        "rank-1 retrieval caps precision@3 at 33% and precision@5 at 20% - there just aren't 3 or 5 "
        "correct papers in the library for that query. hit@1 above is a truer read on retrieval quality "
        "here; precision@3/@5 climb accordingly on the multi-answer queries below.\n"
    )

    lines.append("| Query | Relevant paper key(s) | Retrieved top-5 | Hit@1 | P@3 | P@5 |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        relevant_str = ", ".join(r["relevant_paper_keys"])
        retrieved_str = ", ".join(r["retrieved_at_5"])
        hit1_str = "yes" if r["hit_at_1"] else "no"
        lines.append(
            f"| {r['query']} | {relevant_str} | {retrieved_str} | {hit1_str} | "
            f"{r['precision_at_3']:.2f} | {r['precision_at_5']:.2f} |"
        )

    out_path.write_text("\n".join(lines) + "\n")


async def main_async(data_dir: str) -> None:
    search_engine = SemanticSearchEngine(data_dir=data_dir)
    if not search_engine.load_index():
        raise RuntimeError(
            f"No persisted index found in {data_dir}/. Run `python scripts/reindex.py` first."
        )

    queries = json.loads(QUERY_SET_PATH.read_text())
    rows = await evaluate(search_engine, queries)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(rows, RESULTS_PATH)

    n = len(rows)
    print(f"Mean precision@3: {sum(r['precision_at_3'] for r in rows) / n:.1%}")
    print(f"Mean precision@5: {sum(r['precision_at_5'] for r in rows) / n:.1%}")
    print(f"Wrote {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate document-level FAISS retrieval precision@k.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory holding the persisted index")
    args = parser.parse_args()
    asyncio.run(main_async(args.data_dir))


if __name__ == "__main__":
    main()
