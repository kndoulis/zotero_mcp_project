# ABOUTME: Unit tests for the retrieval evaluation harness's precision@k scoring logic.
# ABOUTME: Uses a mocked search engine so these tests exercise no real FAISS index or model.
from unittest.mock import AsyncMock, MagicMock

import pytest

from eval.retrieval_eval import evaluate, write_markdown_report


def make_paper(key: str):
    paper = MagicMock()
    paper.key = key
    return paper


@pytest.mark.asyncio
async def test_evaluate_computes_precision_at_3_and_5():
    engine = MagicMock()
    # Top-5 retrieved: 2 of the top-3 are relevant, 3 of the top-5 are relevant
    engine.search_by_topic = AsyncMock(return_value=[
        (make_paper("A"), 0.9),
        (make_paper("B"), 0.8),
        (make_paper("C"), 0.7),
        (make_paper("D"), 0.6),
        (make_paper("E"), 0.5),
    ])

    queries = [{"query": "test query", "relevant_paper_keys": ["A", "C", "E"]}]

    rows = await evaluate(engine, queries)

    assert len(rows) == 1
    row = rows[0]
    assert row["precision_at_3"] == pytest.approx(2 / 3)  # A, C hit within top-3
    assert row["precision_at_5"] == pytest.approx(3 / 5)  # A, C, E hit within top-5
    assert row["hit_at_1"] == 1.0  # A (relevant) is ranked first
    engine.search_by_topic.assert_awaited_once_with("test query", max_results=5, min_similarity=0.0)


@pytest.mark.asyncio
async def test_evaluate_zero_precision_when_nothing_relevant_retrieved():
    engine = MagicMock()
    engine.search_by_topic = AsyncMock(return_value=[(make_paper("X"), 0.9), (make_paper("Y"), 0.8)])

    queries = [{"query": "test query", "relevant_paper_keys": ["Z"]}]

    rows = await evaluate(engine, queries)

    assert rows[0]["precision_at_3"] == 0.0
    assert rows[0]["precision_at_5"] == 0.0
    assert rows[0]["hit_at_1"] == 0.0


def test_write_markdown_report_includes_mean_and_table(tmp_path):
    rows = [
        {
            "query": "q1",
            "relevant_paper_keys": ["A"],
            "retrieved_at_5": ["A", "B", "C", "D", "E"],
            "precision_at_3": 1 / 3,
            "precision_at_5": 1 / 5,
            "hit_at_1": 1.0,
        },
        {
            "query": "q2",
            "relevant_paper_keys": ["B", "C"],
            "retrieved_at_5": ["B", "C", "D", "E", "F"],
            "precision_at_3": 2 / 3,
            "precision_at_5": 2 / 5,
            "hit_at_1": 1.0,
        },
    ]
    out_path = tmp_path / "retrieval_eval.md"

    write_markdown_report(rows, out_path)

    text = out_path.read_text()
    assert "q1" in text and "q2" in text
    assert "Mean precision@3" in text
    assert "Mean precision@5" in text
