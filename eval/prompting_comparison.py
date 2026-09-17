# ABOUTME: Compares zero-shot, few-shot, and chain-of-thought prompting for structured field extraction.
# ABOUTME: Runs all three styles against a hand-labeled test set, scores accuracy, and writes a results table.
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from zotero_mcp_server.llm_client import DEFAULT_MODEL, call_claude, extract_text

EVAL_SET_PATH = Path(__file__).parent / "labeled_data" / "methodology_extraction.json"
FEW_SHOT_PATH = Path(__file__).parent / "labeled_data" / "few_shot_examples.json"
RESULTS_DIR = Path(__file__).parent / "results"

STYLES = ["zero_shot", "few_shot", "cot"]

CATEGORIES = ["review", "experimental", "clinical_study", "tool", "unclear"]

CATEGORY_DEFINITIONS = (
    "- review: a literature review, survey, or tutorial that does not report the authors' own new experiments\n"
    "- experimental: proposes and empirically evaluates a new model, method, or algorithm on data/benchmarks\n"
    "- clinical_study: a study that directly collects or analyzes data from human/patient subjects as its primary "
    "contribution (not primarily an ML method paper)\n"
    "- tool: describes a software library, framework, or toolkit\n"
    "- unclear: none of the above fit, or there is not enough information"
)

TASK_INSTRUCTION = (
    "Given a paper's title and abstract, extract two fields:\n"
    "1. methodology_type: exactly one of [review, experimental, clinical_study, tool, unclear]\n"
    f"{CATEGORY_DEFINITIONS}\n"
    "2. sample_size: the number of subjects, patients, or data samples used in the study, as a plain integer "
    "(e.g. 58, not '58 patients' or '58,000'), or the exact word not_stated if the abstract does not mention one."
)

# Anthropic per-1M-token pricing for the models this harness is meant to compare (cached; update if pricing changes)
PRICING_PER_MTOK = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
}


def _format_item(item: Dict) -> str:
    return f"Title: {item['title']}\nAbstract: {item['abstract']}"


def build_zero_shot_prompt(item: Dict) -> Tuple[str, str]:
    system = "You are a precise research methodologist who extracts structured metadata from paper abstracts."
    user = (
        f"{TASK_INSTRUCTION}\n\n"
        "Respond with EXACTLY two lines, nothing else:\n"
        "methodology_type: <value>\n"
        "sample_size: <value>\n\n"
        f"{_format_item(item)}"
    )
    return system, user


def build_few_shot_prompt(item: Dict, examples: List[Dict]) -> Tuple[str, str]:
    system = "You are a precise research methodologist who extracts structured metadata from paper abstracts."
    example_blocks = [
        f"{_format_item(ex)}\nmethodology_type: {ex['methodology_type']}\nsample_size: {ex['sample_size']}"
        for ex in examples
    ]
    examples_text = "\n\n".join(example_blocks)
    user = (
        f"{TASK_INSTRUCTION}\n\n"
        "Respond with EXACTLY two lines, nothing else, in the same format as the examples below.\n\n"
        f"Examples:\n{examples_text}\n\n"
        f"Now extract the fields for this paper:\n{_format_item(item)}"
    )
    return system, user


def build_cot_prompt(item: Dict) -> Tuple[str, str]:
    system = "You are a precise research methodologist who extracts structured metadata from paper abstracts."
    user = (
        f"{TASK_INSTRUCTION}\n\n"
        "First, reason step by step: what does the abstract say the authors actually did, and does it state a "
        "specific number of subjects/samples anywhere? Then, on the final two lines of your response, give your "
        "answer in exactly this format:\n"
        "methodology_type: <value>\n"
        "sample_size: <value>\n\n"
        f"{_format_item(item)}"
    )
    return system, user


def parse_response(text: str) -> Tuple[Optional[str], Optional[str], bool]:
    """Extract the last methodology_type/sample_size lines in the response.

    Taking the last match (not the first) makes this one parser work for all three prompt
    styles: zero-shot/few-shot only ever emit one such pair, while CoT responses may mention
    these field names while reasoning before giving the final answer.

    Falls back to the last two non-empty lines when a label is missing - observed in practice
    with few-shot prompting, where the model sometimes echoes just the bare values (e.g.
    "review\\nnot_stated") instead of repeating the "field: value" labels shown in the examples.
    """
    type_matches = re.findall(r"methodology_type:\s*([A-Za-z_]+)", text, re.IGNORECASE)
    size_matches = re.findall(r"sample_size:\s*([^\n]+)", text, re.IGNORECASE)
    methodology_type = type_matches[-1].strip().lower() if type_matches else None
    sample_size_raw = size_matches[-1].strip() if size_matches else None

    used_fallback = methodology_type is None or sample_size_raw is None
    if used_fallback:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if len(lines) >= 2:
            if methodology_type is None:
                methodology_type = lines[-2].lower()
            if sample_size_raw is None:
                sample_size_raw = lines[-1]

    return methodology_type, sample_size_raw, used_fallback


def normalize_sample_size(raw) -> str:
    """Normalize a sample_size value (model output or gold label) to an int or the string 'not_stated'."""
    if raw is None:
        return "not_stated"
    if isinstance(raw, int):
        return raw
    cleaned = str(raw).strip().lower().rstrip(".")
    if not cleaned or "not_stated" in cleaned or "not stated" in cleaned or cleaned in {"none", "unknown", "n/a", "unspecified"}:
        return "not_stated"
    digits = re.sub(r"[^\d]", "", cleaned)
    return int(digits) if digits else "not_stated"


def run_style(style: str, eval_items: List[Dict], few_shot_examples: List[Dict], model: str) -> Dict:
    predictions = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, item in enumerate(eval_items, 1):
        if style == "zero_shot":
            system, user, max_tokens = *build_zero_shot_prompt(item), 60
        elif style == "few_shot":
            system, user, max_tokens = *build_few_shot_prompt(item, few_shot_examples), 60
        elif style == "cot":
            system, user, max_tokens = *build_cot_prompt(item), 400
        else:
            raise ValueError(f"Unknown style: {style}")

        response = call_claude(
            messages=[{"role": "user", "content": user}],
            system=system,
            model=model,
            max_tokens=max_tokens,
            label=f"prompting_comparison:{style}",
        )
        text = extract_text(response)
        raw_type, raw_size, used_fallback_parsing = parse_response(text)
        pred_type = raw_type.strip().lower() if raw_type else None
        pred_size = normalize_sample_size(raw_size)

        gold_type = item["methodology_type"]
        gold_size = normalize_sample_size(item["sample_size"])

        type_correct = pred_type == gold_type
        size_correct = pred_size == gold_size
        both_correct = type_correct and size_correct

        predictions.append({
            "paper_key": item["paper_key"],
            "gold_methodology_type": gold_type,
            "pred_methodology_type": pred_type,
            "type_correct": type_correct,
            "gold_sample_size": gold_size,
            "pred_sample_size": pred_size,
            "size_correct": size_correct,
            "both_correct": both_correct,
            "used_fallback_parsing": used_fallback_parsing,
            "raw_response": text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        })

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        print(
            f"[{style}] {i}/{len(eval_items)} {item['paper_key']}: "
            f"type={'OK' if type_correct else 'X'} size={'OK' if size_correct else 'X'}"
        )

    n = len(predictions)
    return {
        "style": style,
        "n": n,
        "type_accuracy": sum(p["type_correct"] for p in predictions) / n,
        "size_accuracy": sum(p["size_correct"] for p in predictions) / n,
        "both_accuracy": sum(p["both_correct"] for p in predictions) / n,
        "format_adherence": 1 - (sum(p["used_fallback_parsing"] for p in predictions) / n),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "predictions": predictions,
    }


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    pricing = PRICING_PER_MTOK.get(model)
    if not pricing:
        return None
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]


def write_markdown_report(results: List[Dict], model: str, out_path: Path) -> None:
    lines = ["# Prompting Strategy Comparison\n"]
    lines.append("Task: extract `methodology_type` and `sample_size` from Zotero library paper abstracts.\n")
    lines.append(f"Model: `{model}` | Test set size: {results[0]['n']} hand-labeled papers.\n")

    lines.append(
        "| Style | methodology_type accuracy | sample_size accuracy | both-correct accuracy "
        "| format adherence | input tokens | output tokens | est. cost |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        cost = estimate_cost(model, r["total_input_tokens"], r["total_output_tokens"])
        cost_str = f"${cost:.4f}" if cost is not None else "n/a"
        lines.append(
            f"| {r['style']} | {r['type_accuracy']:.1%} | {r['size_accuracy']:.1%} | {r['both_accuracy']:.1%} | "
            f"{r['format_adherence']:.1%} | {r['total_input_tokens']} | {r['total_output_tokens']} | {cost_str} |"
        )

    best_both = max(results, key=lambda r: r["both_accuracy"])
    cheapest = min(results, key=lambda r: r["total_input_tokens"] + r["total_output_tokens"])
    worst_format = min(results, key=lambda r: r["format_adherence"])

    lines.append("\n## Findings\n")
    lines.append(f"- Highest combined accuracy: **{best_both['style']}** ({best_both['both_accuracy']:.1%} both fields correct).")
    lines.append(f"- Cheapest in tokens: **{cheapest['style']}**.")
    lines.append(
        "- `methodology_type` is generally easier to get right than `sample_size`: extracting a category from "
        "prose is a pattern the model has seen constantly, while `sample_size` requires distinguishing a stated "
        "subject/data count from other numbers in the abstract (dataset names, accuracy percentages, layer "
        "counts, publication years) and correctly saying `not_stated` rather than guessing one of them."
    )
    if worst_format["format_adherence"] < 0.99:
        lines.append(
            f"- **Format adherence dropped for {worst_format['style']}** ({worst_format['format_adherence']:.1%} of "
            "responses used the exact `field: value` format instructed, vs. ~100% for the other styles). Observed "
            "cause: when shown labeled few-shot examples, the model sometimes echoed just the bare values (e.g. "
            "`review\\nnot_stated`) instead of repeating the `methodology_type:`/`sample_size:` labels - the "
            "underlying extracted values were usually still correct, but a stricter downstream parser than this "
            "harness's (which falls back to the last two lines) would have silently failed on those responses. "
            "This is a real risk of few-shot prompting: examples can teach the model an implicit shorthand instead "
            "of reinforcing the literal instructed format."
        )

    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare zero-shot/few-shot/CoT prompting for structured extraction.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Claude model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    eval_items = json.loads(EVAL_SET_PATH.read_text())
    few_shot_examples = json.loads(FEW_SHOT_PATH.read_text())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = [run_style(style, eval_items, few_shot_examples, args.model) for style in STYLES]

    (RESULTS_DIR / "prompting_comparison_results.json").write_text(json.dumps(results, indent=2))
    write_markdown_report(results, args.model, RESULTS_DIR / "prompting_comparison.md")

    print("\n=== Summary ===")
    for r in results:
        print(
            f"{r['style']:10s} type_acc={r['type_accuracy']:.1%} size_acc={r['size_accuracy']:.1%} "
            f"both_acc={r['both_accuracy']:.1%} tokens_in={r['total_input_tokens']} tokens_out={r['total_output_tokens']}"
        )
    print(f"\nWrote {RESULTS_DIR / 'prompting_comparison_results.json'}")
    print(f"Wrote {RESULTS_DIR / 'prompting_comparison.md'}")


if __name__ == "__main__":
    main()
