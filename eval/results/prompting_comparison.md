# Prompting Strategy Comparison

Task: extract `methodology_type` and `sample_size` from Zotero library paper abstracts.

Model: `claude-haiku-4-5-20251001` | Test set size: 26 hand-labeled papers.

| Style | methodology_type accuracy | sample_size accuracy | both-correct accuracy | format adherence | input tokens | output tokens | est. cost |
|---|---|---|---|---|---|---|---|
| zero_shot | 96.2% | 96.2% | 92.3% | 100.0% | 15201 | 423 | $0.0173 |
| few_shot | 100.0% | 96.2% | 96.2% | 0.0% | 49911 | 208 | $0.0510 |
| cot | 88.5% | 96.2% | 88.5% | 96.2% | 16189 | 7522 | $0.0538 |

## Findings

- Highest combined accuracy: **few_shot** (96.2% both fields correct).
- Cheapest in tokens: **zero_shot**.
- `methodology_type` is generally easier to get right than `sample_size`: extracting a category from prose is a pattern the model has seen constantly, while `sample_size` requires distinguishing a stated subject/data count from other numbers in the abstract (dataset names, accuracy percentages, layer counts, publication years) and correctly saying `not_stated` rather than guessing one of them.
- **Format adherence dropped for few_shot** (0.0% of responses used the exact `field: value` format instructed, vs. ~100% for the other styles). Observed cause: when shown labeled few-shot examples, the model sometimes echoed just the bare values (e.g. `review\nnot_stated`) instead of repeating the `methodology_type:`/`sample_size:` labels - the underlying extracted values were usually still correct, but a stricter downstream parser than this harness's (which falls back to the last two lines) would have silently failed on those responses. This is a real risk of few-shot prompting: examples can teach the model an implicit shorthand instead of reinforcing the literal instructed format.
