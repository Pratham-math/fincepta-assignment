# Filing Fact-Checker

A multi-agent system that checks statements about CrowdStrike Holdings against its official SEC filings (10-K and 10-Q), returning a verdict with exact evidence citations.

## Verdicts

| Verdict | Meaning |
|---|---|
| `SUPPORTED` | Filings confirm the statement |
| `CONTRADICTED` | Filings show the statement is wrong |
| `OUTDATED` | An earlier filing supports it, but a later filing changes the picture |
| `NOT_ENOUGH_INFO` | Filings don't say either way |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama
- Download from https://ollama.com
- Pull the model: `ollama pull qwen2.5:7b`
- Ollama starts automatically as a background service

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env if needed (e.g. different model or port)
```

### 4. Ingest the filings
```bash
python ingest.py
```
This extracts text from all three PDFs and builds `data/index.json`.

### 5. Run the API
```bash
uvicorn src.api:app --reload
```

### 6. Run the UI
```bash
streamlit run src/ui.py
```

## Filings

| Filing | Period | File |
|---|---|---|
| 10-K | FY ended Jan 31, 2024 | crwd-20240131.pdf |
| 10-Q | Quarter ended Jul 31, 2024 | 0001535527-24-000020.pdf |
| 10-Q | Quarter ended Oct 31, 2024 | crwd-20241031.pdf |

## Architecture

```
Streamlit UI → FastAPI (streaming) → LangGraph
                                        ├── Decomposer node
                                        ├── Memory recall
                                        └── Executor node ↔ Filing tools (BM25 search)
                                                            ↓
                                                    Citation validator → verdict
```

## Evaluation

```bash
python evaluate.py
```
Runs all 20 test statements in 3 configurations (full system, no memory, no decomposer) and saves results to `predictions.json`.

## Design choices vs MERMAID paper

- **Kept**: Decomposer + Executor separation, reusable evidence memory, extensible toolset
- **Simplified**: SQLite instead of a knowledge graph; BM25 instead of web search; page-level chunking instead of sentence-level
- **Dropped**: Web tools, MCP servers, parallel workers — not needed for a fixed 3-filing corpus
