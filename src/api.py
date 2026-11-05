"""
api.py — FastAPI streaming endpoint for the Filing Fact-Checker.

Endpoints:
  POST /check        — stream agent steps + final result
  POST /check/batch  — run multiple statements sequentially (streams all)
  GET  /health       — liveness check

Run with:
  uvicorn src.api:app --reload --port 8000
"""

import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent import run_check_stream

app = FastAPI(title="Filing Fact-Checker", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class CheckRequest(BaseModel):
    statement: str
    use_decomposer: bool = True


class BatchRequest(BaseModel):
    statements: list[str]
    use_decomposer: bool = True


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------
async def _stream_check(statement: str, use_decomposer: bool) -> AsyncGenerator[str, None]:
    """Yield newline-delimited JSON events from the agent."""
    try:
        for event in run_check_stream(statement, use_decomposer=use_decomposer):
            yield json.dumps(event, ensure_ascii=False) + "\n"
    except Exception as e:
        error_event = {
            "type": "error",
            "detail": str(e),
        }
        yield json.dumps(error_event) + "\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check")
async def check(req: CheckRequest):
    """
    Stream the agent's reasoning steps and final verdict.

    Each line is a JSON object:
      {"type": "step",   "step": N, "action": "tool_call"|"verdict"|"decomposed", "detail": "..."}
      {"type": "result", "data": { verdict, explanation, evidence, usage, finish_reason }}
      {"type": "error",  "detail": "..."}
    """
    if not req.statement or not req.statement.strip():
        raise HTTPException(status_code=400, detail="statement must not be empty")

    return StreamingResponse(
        _stream_check(req.statement.strip(), req.use_decomposer),
        media_type="application/x-ndjson",
    )


@app.post("/check/batch")
async def check_batch(req: BatchRequest):
    """
    Stream checks for multiple statements sequentially.
    Each statement's events are prefixed with {"type": "statement_start", "index": N, "statement": "..."}.
    """
    if not req.statements:
        raise HTTPException(status_code=400, detail="statements list must not be empty")

    async def _batch_stream():
        for i, statement in enumerate(req.statements):
            # Announce which statement we're starting
            yield json.dumps({
                "type": "statement_start",
                "index": i,
                "total": len(req.statements),
                "statement": statement,
            }) + "\n"

            async for chunk in _stream_check(statement.strip(), req.use_decomposer):
                # Tag each event with its index
                try:
                    event = json.loads(chunk.strip())
                    event["index"] = i
                    yield json.dumps(event) + "\n"
                except Exception:
                    yield chunk

    return StreamingResponse(_batch_stream(), media_type="application/x-ndjson")
