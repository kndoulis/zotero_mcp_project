# Zotero MCP Server

A Model Context Protocol (MCP) server that integrates with your Zotero library to provide semantic search, retrieval-augmented generation, and agentic research assistance, backed by a FAISS vector index and the Claude API.

## Architecture Overview

There's one shared retrieval core, consumed by three different surfaces:

```
Zotero Web API
      │
      ▼
ZoteroClient (zotero_client.py)
      │  fetches items + full-text
      ▼
SemanticSearchEngine (semantic_search.py)
  ├─ document-level FAISS index  (title+abstract+full-text preview, one vector per paper)
  ├─ chunk-level FAISS index     (papers split into ~500-token, section-tagged, overlapping chunks)
  ├─ TF-IDF matrix               (keyword search, rebuilt cheaply from the same text on load)
  └─ persisted to data/*.faiss + data/*.json  (scripts/reindex.py builds this; nothing below re-fetches from Zotero)
      │
      ├──────────────────────────┬──────────────────────────────┐
      ▼                          ▼                                ▼
main.py                    rag_generate.py                   agent.py
MCP server, 8 tools         ask_library(): retrieves           standalone CLI, tool-use loop
(search, chapter             chunks, prompts Claude,           over search_papers / summarize_paper /
 suggestions, ask_library,   returns a cited answer             save_note (llm_client.py underneath)
 refresh, ...)                    │
                                   ▼
                              api.py
                              FastAPI wrapper around ask_library,
                              deployable to Cloud Run
```

`llm_client.py` is the shared Claude API wrapper (`rag_generate.py` and `agent.py` both call through it): configurable model (defaults to `claude-haiku-4-5-20251001`, pass `model="claude-sonnet-5"` to compare), and every call logs `input_tokens`/`output_tokens` to `data/usage_log.jsonl` for cost tracking.

`eval/` holds two independent evaluation harnesses (see [Evaluation & Results](#evaluation--results)): a prompting-strategy comparison for a structured-extraction task, and a precision@k evaluation of the document-level retrieval.

## Project Structure

```
src/zotero_mcp_server/
├── __init__.py
├── main.py              # MCP server - 8 tools, entry point: python -m zotero_mcp_server.main
├── zotero_client.py      # Zotero Web API client (auth, pagination, full-text fetch)
├── semantic_search.py    # FAISS document/chunk indexes, TF-IDF, persistence, section extraction
├── llm_client.py         # Shared Anthropic client: model config + token usage logging
├── rag_generate.py       # ask_library(): retrieve-then-generate with citations
├── agent.py              # Standalone tool-use agent, entry point: python -m zotero_mcp_server.agent
└── api.py                # FastAPI wrapper (POST /ask, GET /health), entry point: python -m zotero_mcp_server.api

scripts/
└── reindex.py             # Build/refresh the persisted index without an MCP client attached

eval/
├── labeled_data/          # Hand-labeled test sets (drawn from the real library)
├── results/                # Generated reports (committed - these are deliverables, not cache)
├── prompting_comparison.py # Zero-shot vs few-shot vs CoT comparison for structured extraction
└── retrieval_eval.py       # precision@3 / precision@5 for search_by_topic

tests/                     # pytest, one file per module above (+ test_prompting_comparison.py, test_retrieval_eval.py)
data/                      # gitignored: FAISS indexes, usage_log.jsonl, agent_notes.txt
Dockerfile                 # Bakes in data/ for Cloud Run deployment of api.py
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd zotero_mcp_project
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up credentials:
```bash
cp .env.example .env
# Edit .env: ZOTERO_API_KEY, ZOTERO_USER_ID, ANTHROPIC_API_KEY
```

## Getting Your Credentials

- **Zotero API Key**: https://www.zotero.org/settings/keys — create a private key with read access to your library
- **Zotero User ID**: found in your Zotero profile URL (`https://www.zotero.org/users/YOUR_USER_ID`)
- **Anthropic API Key**: https://console.anthropic.com/settings/keys — required for `ask_library`, the standalone agent, the FastAPI wrapper, and both eval harnesses; not required for plain search/keyword tools

## Usage

### 1. Build the search index

Most of the system reads a *persisted* index rather than hitting the Zotero API live. Build (or refresh) it first:

```bash
python scripts/reindex.py              # full re-fetch + re-index
python scripts/reindex.py --no-refresh # load the existing persisted index instead, if present
```

This prints a summary (papers fetched, document/chunk index sizes, section-title distribution) so you can sanity-check indexing without needing an MCP client connected.

### 2. Run the MCP server

```bash
python -m zotero_mcp_server.main
```

Connect it to Claude Desktop by adding to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "zotero": {
      "command": "python",
      "args": ["-m", "zotero_mcp_server.main"],
      "cwd": "/path/to/zotero_mcp_project"
    }
  }
}
```

The server initializes lazily on first tool call (loading the persisted index if present, or building one). `refresh_library_index` forces a rebuild from Zotero.

#### Available Tools

| Tool | Purpose | Key parameters |
|---|---|---|
| `search_papers_by_topic` | Semantic search over the document-level FAISS index | `topic`, `max_results`, `min_similarity` |
| `search_papers_by_keywords` | TF-IDF keyword search | `keywords`, `max_results` |
| `find_relevant_sections` | Find relevant sections within one paper | `paper_key`, `topic`, `max_sections` |
| `get_library_overview` | Library stats + collections | `include_collections` |
| `get_paper_details` | Full metadata for one paper | `paper_key`, `include_fulltext` |
| `suggest_papers_for_chapter` | Paper + section suggestions for a chapter you're writing | `chapter_topic`, `chapter_outline`, `max_papers` |
| `refresh_library_index` | Force a full re-fetch + re-index from Zotero | — |
| `ask_library` | Retrieve chunks + Claude-generated cited answer (RAG) | `query`, `max_chunks`, `model` |

### 3. Ask questions with citations (RAG)

Via the MCP tool above, or directly:
```python
from zotero_mcp_server.semantic_search import SemanticSearchEngine
from zotero_mcp_server.rag_generate import ask_library

engine = SemanticSearchEngine()
engine.load_index()
result = await ask_library("What GAN techniques are used for medical imaging data augmentation?", engine)
print(result["answer"])     # cited answer, e.g. "...uses CycleGAN [UU683AUC]..."
print(result["sources"])    # paper_key, title, section_title, relevance per source used
```

### 4. Run the standalone agent

Separate from the MCP server - a CLI tool-use loop with `search_papers`, `summarize_paper`, and `save_note` tools, an 8-turn iteration cap, and every step printed live:

```bash
python -m zotero_mcp_server.agent "find papers on GANs for medical imaging and summarize their methodology"
python -m zotero_mcp_server.agent "..." --model claude-sonnet-5 --max-iterations 5
```

### 5. Run the RAG API / deploy to Cloud Run

See the [Cloud Deployment](#cloud-deployment-rag-api) section below.

## Evaluation & Results

Both harnesses are pure Python + the Claude API (prompting comparison) or pure FAISS (retrieval eval) - run them with `python eval/<script>.py`, full output in `eval/results/`.

### Prompting Strategy Comparison

Task: extract `methodology_type` and `sample_size` from 26 hand-labeled paper abstracts, compared across zero-shot / few-shot / chain-of-thought prompting on `claude-haiku-4-5-20251001`. Full report: [`eval/results/prompting_comparison.md`](eval/results/prompting_comparison.md).

| Style | methodology_type acc. | sample_size acc. | both-correct | format adherence | est. cost |
|---|---|---|---|---|---|
| zero-shot | 96.2% | 96.2% | 92.3% | 100.0% | $0.017 |
| few-shot | 100.0% | 96.2% | 96.2% | 0.0%* | $0.051 |
| chain-of-thought | 88.5% | 96.2% | 88.5% | 96.2% | $0.054 |

\* Genuine finding, not a bug: with few-shot examples shown in `field: value` format, Haiku consistently echoed back just the bare values (e.g. `review\nnot_stated`) instead of repeating the labels - the extracted values were almost always still correct, but a stricter downstream parser than this harness's (which falls back to the last two lines of the response) would have silently failed on every one of those responses. A concrete illustration of a real few-shot prompting risk: examples can teach an implicit shorthand instead of reinforcing the literal instructed output format.

Run it: `python eval/prompting_comparison.py` (optional `--model claude-sonnet-5`).

### Retrieval Evaluation

15 hand-written queries with known relevant paper(s), evaluated against `search_by_topic` (the document-level FAISS index). Full report: [`eval/results/retrieval_eval.md`](eval/results/retrieval_eval.md).

**Mean precision@3: 44.4% · Mean precision@5: 26.7% · Mean hit@1: 100.0%**

Precision@k is mechanically bounded by `|relevant papers| / k`: 11 of the 15 queries have exactly one relevant paper in the library, so even a perfect rank-1 retrieval caps precision@3 at 33% and precision@5 at 20% for those queries - there just aren't 3 or 5 correct answers to find. hit@1 (was the correct paper ranked first) is the more meaningful number here, and it's 100% across all 15 queries; precision@3/@5 climb accordingly on the multi-answer queries in the full table.

Run it: `python eval/retrieval_eval.py`.

## Cloud Deployment (RAG API)

The `ask_library` pipeline is also available as a standalone HTTP API (`src/zotero_mcp_server/api.py`), separate from the MCP server, with a single `POST /ask` endpoint. It's containerized and deployable to Google Cloud Run's free tier.

**Important:** the API loads a *persisted* search index at startup - it does not fetch from Zotero at request time. Build the index locally first:

```bash
python scripts/reindex.py
```

### Run locally

```bash
python -m zotero_mcp_server.api          # reads data/ built above, serves on :8080
curl -X POST http://localhost:8080/ask -H "Content-Type: application/json" \
     -d '{"query": "What GAN techniques are used for data augmentation?"}'
```

### Run with Docker

The image bakes in whatever is currently in `data/` at build time, so re-run `scripts/reindex.py` and rebuild whenever you want the deployed API to reflect a refreshed library.

```bash
docker build -t zotero-rag-api .
docker run -p 8080:8080 -e ANTHROPIC_API_KEY=your_key_here zotero-rag-api
```

### Deploy to Google Cloud Run

Requires the [gcloud CLI](https://cloud.google.com/sdk/docs/install) authenticated against a GCP project with billing enabled (Cloud Run's free tier covers light traffic).

```bash
gcloud run deploy zotero-rag-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ANTHROPIC_API_KEY=your_key_here \
  --memory 2Gi
```

`--source .` builds the `Dockerfile` in Cloud Build and deploys directly - no separate image push step needed. `--memory 2Gi` is a safety margin for the sentence-transformers model load at startup; scale down once you've confirmed actual usage. No Zotero credentials are needed in production since the index is baked into the image, not fetched live - only `ANTHROPIC_API_KEY` is required, and it's passed as a runtime env var, never baked into the image or committed to source.

### API reference

`POST /ask`
```json
{"query": "your question", "max_chunks": 8, "model": "claude-haiku-4-5-20251001"}
```
→
```json
{"answer": "...", "sources": [{"paper_key": "...", "title": "...", "section_title": "...", "relevance": 0.87}]}
```

`GET /health` → `{"status": "ok", "papers_indexed": 98}`

## Configuration

Environment variables (see `.env.example`):

- `ZOTERO_API_KEY`, `ZOTERO_USER_ID` — required for `scripts/reindex.py` and the MCP server's `refresh_library_index`
- `ANTHROPIC_API_KEY` — required for `ask_library`, `agent.py`, `api.py`, and both eval harnesses
- Model choice is a function parameter, not an env var: everything defaults to `claude-haiku-4-5-20251001` and accepts `model="claude-sonnet-5"` (or any other model ID) as an override, per call

## Development

### Running Tests

```bash
pytest tests/
```

### Cost Tracking

Every Claude API call (RAG generation, agent tool use, prompting comparison) appends one line to `data/usage_log.jsonl`:
```json
{"timestamp": "...", "label": "ask_library", "model": "claude-haiku-4-5-20251001", "input_tokens": 1863, "output_tokens": 337}
```

## Performance Notes

- Initial indexing may take a few minutes depending on library size; subsequent runs load the persisted index instead of rebuilding
- Embeddings are computed locally using sentence-transformers; only RAG generation and the agent call out to the Claude API
- Chunking only splits papers that have `full_text` populated by Zotero's fulltext extraction; papers without it get a single title+abstract chunk

## Requirements

- Python 3.9+
- Zotero account with API access
- Anthropic API key
- Internet connection for initial setup and API calls

## License

MIT License - see LICENSE file for details.
