"""
memory.py — SQLite-backed evidence memory for the Filing Fact-Checker.

Stores previously found passages with one-line summaries so related
statements can reuse evidence without repeating BM25 searches.

Schema:
    evidence(
        id           INTEGER PRIMARY KEY,
        filing_id    TEXT,
        page_number  INTEGER,
        passage      TEXT,
        section      TEXT,
        summary      TEXT,      -- one-line LLM-generated summary
        query_terms  TEXT,      -- space-separated terms from the statement
        filing_date  TEXT,
        created_at   TEXT
    )

Usage:
    from src.memory import recall, store, get_stats, reset_session_stats
"""

import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------
_DB_FILE = Path(os.getenv("MEMORY_DB", "data/memory.db"))
_DB_FILE.parent.mkdir(parents=True, exist_ok=True)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS evidence (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id   TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    passage     TEXT NOT NULL,
    section     TEXT,
    summary     TEXT NOT NULL,
    query_terms TEXT,
    filing_date TEXT,
    created_at  TEXT NOT NULL,
    UNIQUE(filing_id, page_number, passage)
)
"""

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_FILE))
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_SQL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Session-level stats (reset each run_check call)
# ---------------------------------------------------------------------------
_session_hits = 0

def reset_session_stats() -> None:
    global _session_hits
    _session_hits = 0

def get_stats() -> dict:
    conn = _get_conn()
    count = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    conn.close()
    return {"stored_passages": count, "session_hits": _session_hits}


# ---------------------------------------------------------------------------
# Tokenizer (same as retrieval.py for consistency)
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------
def _generate_summary(passage: str, filing_id: str, page_number: int) -> str:
    """
    Generate a one-line summary of a filing passage using the LLM.
    Falls back to a truncated passage if the LLM fails.
    """
    try:
        from src.llm import call_llm
        messages = [
            {
                "role": "user",
                "content": (
                    f"Summarize this SEC filing passage in one short sentence (max 20 words). "
                    f"Include the key metric and value if present. "
                    f"Reply with ONLY the summary sentence, nothing else.\n\n"
                    f"Filing: {filing_id}, Page {page_number}\n"
                    f"Passage: {passage[:500]}"
                ),
            }
        ]
        response = call_llm(messages=messages, temperature=0.0)
        summary = (response["content"] or "").strip()
        # Strip quotes if model wrapped in them
        summary = summary.strip('"').strip("'").strip()
        if summary and len(summary) > 5:
            return summary[:200]
    except Exception:
        pass

    # Fallback: first 100 chars of passage
    return passage[:100].replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
def store(evidence_list: list, statement: str) -> int:
    """
    Store cited evidence passages from a successful verdict.
    Generates a one-line summary for each passage.
    Deduplicates by (filing_id, page_number, passage).

    Args:
        evidence_list: list of Evidence objects or dicts with
                       filing_id, page_number, passage, section
        statement:     the original claim (used to extract query_terms)

    Returns:
        Number of new rows inserted.
    """
    if not evidence_list:
        return 0

    # Extract query terms from the statement for future BM25 recall
    query_terms = " ".join(_tokenize(statement)[:20])
    now = datetime.utcnow().isoformat()

    conn = _get_conn()
    inserted = 0

    for ev in evidence_list:
        # Accept both Evidence objects and plain dicts
        if hasattr(ev, "filing_id"):
            filing_id   = ev.filing_id
            page_number = ev.page_number
            passage     = ev.passage
            section     = ev.section
        else:
            filing_id   = ev.get("filing_id", "unknown")
            page_number = ev.get("page_number", 0)
            passage     = ev.get("passage", "")
            section     = ev.get("section")

        if not passage or len(passage.strip()) < 10:
            continue

        # Get filing_date from index if available
        filing_date = _get_filing_date(filing_id)

        # Generate summary
        summary = _generate_summary(passage, filing_id, page_number)

        try:
            conn.execute(
                """INSERT OR IGNORE INTO evidence
                   (filing_id, page_number, passage, section, summary,
                    query_terms, filing_date, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (filing_id, page_number, passage, section, summary,
                 query_terms, filing_date, now),
            )
            if conn.total_changes > 0:
                inserted += 1
        except sqlite3.Error:
            continue

    conn.commit()
    conn.close()
    return inserted


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------
def recall(statement: str, top_k: int = 3) -> list[dict]:
    """
    Search stored evidence for passages relevant to this statement.
    Uses BM25 over stored summaries + passages.

    Args:
        statement: the claim being verified
        top_k:     max number of passages to return (default 3)

    Returns:
        List of dicts with filing_id, page_number, passage, section,
        summary, filing_date, score.
        Empty list if memory has fewer than 2 entries.
    """
    global _session_hits

    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM evidence ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    if len(rows) < 2:
        # Not enough memory to be useful
        return []
    # For small memory stores, use simple keyword overlap scoring
    # BM25 IDF collapses when corpus is tiny (all terms appear in all docs)
    query_tokens = set(_tokenize(statement))

    results = []
    for row in rows:
        doc_tokens = set(_tokenize(f"{row['summary']} {row['passage']}"))
        overlap = len(query_tokens & doc_tokens)
        if overlap > 0:
            score = overlap / max(len(query_tokens), 1)
            results.append({
                "filing_id":   row["filing_id"],
                "page_number": row["page_number"],
                "passage":     row["passage"],
                "section":     row["section"],
                "summary":     row["summary"],
                "filing_date": row["filing_date"],
                "score":       round(score, 4),
            })

    # Sort by score descending, take top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_k]

    if results:
        _session_hits += len(results)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_filing_dates: dict[str, str] = {}

def _get_filing_date(filing_id: str) -> str:
    """Look up the filing date from the index (cached)."""
    global _filing_dates
    if not _filing_dates:
        try:
            import json
            index_file = Path(os.getenv("INDEX_FILE", "data/index.json"))
            with open(index_file, encoding="utf-8") as f:
                pages = json.load(f)
            for p in pages:
                fid = p.get("filing_id", "")
                if fid not in _filing_dates:
                    _filing_dates[fid] = p.get("filing_date", "")
        except Exception:
            pass
    return _filing_dates.get(filing_id, "")


def clear_memory() -> None:
    """Delete all stored evidence. Useful for ablation testing."""
    conn = _get_conn()
    conn.execute("DELETE FROM evidence")
    conn.commit()
    conn.close()
    print("[memory] Cleared all stored evidence.")
