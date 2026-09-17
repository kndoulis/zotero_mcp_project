# ABOUTME: Unit tests for the prompting-strategy comparison harness's parsing, normalization, and scoring logic.
# ABOUTME: Uses a mocked Claude call for run_style so these tests make no real API calls.
from unittest.mock import MagicMock, patch

from eval.prompting_comparison import (
    build_few_shot_prompt,
    normalize_sample_size,
    parse_response,
    run_style,
)


def test_parse_response_simple_format():
    text = "methodology_type: experimental\nsample_size: 58"
    mtype, size, used_fallback = parse_response(text)
    assert mtype == "experimental"
    assert size == "58"
    assert used_fallback is False


def test_parse_response_takes_last_match_for_cot_reasoning():
    text = (
        "Let me think. The abstract mentions methodology_type could be experimental, "
        "and sample_size might be 100.\n\n"
        "After reconsidering:\n"
        "methodology_type: review\n"
        "sample_size: not_stated"
    )
    mtype, size, used_fallback = parse_response(text)
    assert mtype == "review"
    assert size == "not_stated"
    assert used_fallback is False


def test_parse_response_missing_fields_returns_none():
    mtype, size, used_fallback = parse_response("I don't know.")
    assert mtype is None
    assert size is None
    assert used_fallback is True


def test_parse_response_falls_back_to_bare_values_when_labels_are_dropped():
    """Observed with few-shot prompting: model echoes just the values, no field-name prefixes."""
    mtype, size, used_fallback = parse_response("review\nnot_stated")
    assert mtype == "review"
    assert size == "not_stated"
    assert used_fallback is True


def test_normalize_sample_size_variants():
    assert normalize_sample_size("58") == 58
    assert normalize_sample_size("100,000") == 100000
    assert normalize_sample_size("over 100,000") == 100000
    assert normalize_sample_size("not_stated") == "not_stated"
    assert normalize_sample_size("not stated") == "not_stated"
    assert normalize_sample_size(None) == "not_stated"
    assert normalize_sample_size(166) == 166
    assert normalize_sample_size("unknown") == "not_stated"


def test_build_few_shot_prompt_includes_examples_and_target():
    examples = [{
        "title": "Example Paper",
        "abstract": "An example abstract.",
        "methodology_type": "review",
        "sample_size": "not_stated",
    }]
    target = {"title": "Target Paper", "abstract": "A target abstract."}

    system, user = build_few_shot_prompt(target, examples)

    assert "Example Paper" in user
    assert "methodology_type: review" in user
    assert "Target Paper" in user
    assert "precise research methodologist" in system.lower()


def test_run_style_scores_predictions_against_gold():
    eval_items = [
        {"paper_key": "P1", "title": "T1", "abstract": "A1", "methodology_type": "review", "sample_size": "not_stated"},
        {"paper_key": "P2", "title": "T2", "abstract": "A2", "methodology_type": "experimental", "sample_size": 58},
    ]

    mock_response_correct = MagicMock()
    mock_response_correct.content = [MagicMock(type="text", text="methodology_type: review\nsample_size: not_stated")]
    mock_response_correct.usage.input_tokens = 50
    mock_response_correct.usage.output_tokens = 10

    mock_response_wrong = MagicMock()
    mock_response_wrong.content = [MagicMock(type="text", text="methodology_type: tool\nsample_size: 12")]
    mock_response_wrong.usage.input_tokens = 55
    mock_response_wrong.usage.output_tokens = 12

    with patch(
        "eval.prompting_comparison.call_claude",
        side_effect=[mock_response_correct, mock_response_wrong],
    ):
        result = run_style("zero_shot", eval_items, few_shot_examples=[], model="claude-haiku-4-5-20251001")

    assert result["n"] == 2
    assert result["type_accuracy"] == 0.5
    assert result["size_accuracy"] == 0.5
    assert result["both_accuracy"] == 0.5
    assert result["total_input_tokens"] == 105
    assert result["total_output_tokens"] == 22
    assert result["predictions"][0]["type_correct"] is True
    assert result["predictions"][1]["type_correct"] is False
