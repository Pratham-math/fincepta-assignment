"""
memory.py — Knowledge-graph memory for the Filing Fact-Checker.

Adapted from MERMAID: stores evidence as entity-relation-value triples
in SQLite. Recall traverses the graph by matching entities and relations
from the new statement against stored triples.

Schema:
    entities(id, name, type)
        e.g. ("CrowdStrike", "company"), ("total_revenue_FY2024", "metric")

    triples(id, subject_id, relation, object_value,
            filing_id, page_number, passage, summary, confidence,
            filing_date, created_at)
        e.g. subject="CrowdStrike", relation="total_revenue_FY2024",
             object_value="3055555000", filing_id="10k_fy2024", page=69

Public API (same interface as before so agent.py needs no changes):
    recall(statement, top_k)  → list of dicts with passage, summary, etc.
    store(evidence_list, statement)  → int (triples inserted)
    get_stats()  → dict
    reset_session_stats()
    clear_memory()
"""

import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------
_DB_FILE = Path(os.getenv("MEMORY_DB", "data/memory.db"))
_DB_FILE.parent.mkdir(parents=True, exist_ok=True)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL UNIQUE,
    type    TEXT NOT NULL DEFAULT 'unknown'
);

CREATE TABLE IF NOT EXISTS triples (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id   INTEGER NOT NULL REFERENCES entities(id),
    relation     TEXT NOT NULL,
    object_value TEXT NOT NULL,
    filing_id    TEXT NOT NULL,
    page_number  INTEGER NOT NULL,
    passage      TEXT NOT NULL,
    summary      TEXT NOT NULL,
    confidence   TEXT NOT NULL DEFAULT 'medium',
    filing_date  TEXT,
    created_at   TEXT NOT NULL,
    UNIQUE(subject_id, relation, filing_id, page_number)
);

CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject_id);
CREATE INDEX IF NOT EXISTS idx_triples_relation ON triples(relation);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_FILE))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Session stats
# ---------------------------------------------------------------------------
_session_hits = 0


def reset_session_stats() -> None:
    global _session_hits
    _session_hits = 0


def get_stats() -> dict:
    conn = _get_conn()
    entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    triples  = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
    conn.close()
    return {
        "stored_entities": entities,
        "stored_triples":  triples,
        "session_hits":    _session_hits,
    }


# ---------------------------------------------------------------------------
# Triple extraction via LLM
# ---------------------------------------------------------------------------
_EXTRACT_PROMPT = """Extract structured facts from this SEC filing passage as JSON triples.

Passage (from {filing_id}, page {page_number}):
{passage}

Return a JSON array of triples. Each triple must have:
- "subject": the main entity (usually "CrowdStrike" or a specific product/person)
- "relation": a snake_case fact name including time period if present
  Examples: total_revenue_FY2024, arr_jan2024, employees_jan2024,
            operating_loss_FY2024, cash_equivalents_oct2024, delta_lawsuit_oct2024
- "value": the specific value or fact (include units)
- "confidence": "high" if the passage directly states this, "medium" if inferred

Return ONLY the JSON array, no other text. Example:
[{{"subject": "CrowdStrike", "relation": "total_revenue_FY2024", "value": "3055555 thousand USD", "confidence": "high"}}]

If no clear facts can be extracted, return an empty array: []"""


def _extract_triples(passage: str, filing_id: str, page_number: int) -> list[dict]:
    """Extract entity-relation-value triples from a passage using the LLM."""
    try:
        from src.llm import call_llm, extract_json
        messages = [{
            "role": "user",
            "content": _EXTRACT_PROMPT.format(
                filing_id=filing_id,
                page_number=page_number,
                passage=passage[:800],
            ),
        }]
        response = call_llm(messages=messages, temperature=0.0)
        content = (response["content"] or "").strip()
        data = extract_json(content)
        if isinstance(data, list):
            return [t for t in data if isinstance(t, dict)
                    and "subject" in t and "relation" in t and "value" in t]
    except Exception:
        pass
    return []


def _generate_summary(passage: str, filing_id: str, page_number: int) -> str:
    """One-line summary of a passage."""
    try:
        from src.llm import call_llm
        messages = [{
            "role": "user",
            "content": (
                f"Summarize this SEC filing passage in one sentence (max 20 words). "
                f"Include the key metric and value. Reply with ONLY the summary.\n\n"
                f"Filing: {filing_id}, Page {page_number}\n"
                f"Passage: {passage[:400]}"
            ),
        }]
        response = call_llm(messages=messages, temperature=0.0)
        summary = (response["content"] or "").strip().strip('"\'')
        if summary and len(summary) > 5:
            return summary[:200]
    except Exception:
        pass
    return passage[:100].replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# Entity management
# ---------------------------------------------------------------------------
def _get_or_create_entity(conn: sqlite3.Connection, name: str, entity_type: str = "unknown") -> int:
    """Return entity id, creating if not exists."""
    row = conn.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    conn.execute("INSERT INTO entities (name, type) VALUES (?, ?)", (name, entity_type))
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _classify_entity_type(name: str) -> str:
    """Guess entity type from name."""
    name_lower = name.lower()
    if any(w in name_lower for w in ["revenue", "arr", "income", "loss", "margin", "cash", "obligations"]):
        return "metric"
    if any(w in name_lower for w in ["crowdstrike", "delta", "microsoft"]):
        return "company"
    if any(w in name_lower for w in ["fy", "fiscal", "quarter", "q1", "q2", "q3", "q4"]):
        return "period"
    if any(w in name_lower for w in ["falcon", "platform", "sensor"]):
        return "product"
    return "unknown"


# ---------------------------------------------------------------------------
# Filing date lookup
# ---------------------------------------------------------------------------
_filing_dates: dict[str, str] = {}


def _get_filing_date(filing_id: str) -> str:
    global _filing_dates
    if not _filing_dates:
        try:
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


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
def store(evidence_list: list, statement: str) -> int:
    """
    Store evidence passages as knowledge graph triples.

    For each passage:
    1. Generate a one-line summary
    2. Extract entity-relation-value triples via LLM
    3. Insert entities and triples into SQLite

    Returns number of new triples inserted.
    """
    if not evidence_list:
        return 0

    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    inserted = 0

    for ev in evidence_list:
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

        filing_date = _get_filing_date(filing_id)
        summary = _generate_summary(passage, filing_id, page_number)
        triples = _extract_triples(passage, filing_id, page_number)

        if not triples:
            # Fallback: create a descriptive triple from key terms in the passage
            # Extract a keyword-based relation from the passage
            passage_lower = passage.lower()
            if "falcon" in passage_lower or "incident" in passage_lower or "crash" in passage_lower:
                relation = f"security_incident_{filing_id[:10]}"
            elif "revenue" in passage_lower:
                relation = f"revenue_{filing_id[:10]}"
            elif "employee" in passage_lower:
                relation = f"employees_{filing_id[:10]}"
            elif "legal" in passage_lower or "lawsuit" in passage_lower:
                relation = f"legal_proceeding_{filing_id[:10]}"
            else:
                relation = f"fact_{filing_id}_p{page_number}"
            triples = [{
                "subject": "CrowdStrike",
                "relation": relation,
                "value": passage[:200],
                "confidence": "medium",
            }]

        for triple in triples:
            subject = triple.get("subject", "CrowdStrike")
            relation = triple.get("relation", "unknown")
            value = str(triple.get("value", ""))
            confidence = triple.get("confidence", "medium")

            entity_type = _classify_entity_type(subject)
            subject_id = _get_or_create_entity(conn, subject, entity_type)

            try:
                conn.execute(
                    """INSERT OR IGNORE INTO triples
                       (subject_id, relation, object_value, filing_id,
                        page_number, passage, summary, confidence,
                        filing_date, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (subject_id, relation, value, filing_id,
                     page_number, passage, summary, confidence,
                     filing_date, now),
                )
                if conn.total_changes > inserted:
                    inserted = conn.total_changes
            except sqlite3.Error:
                continue

    conn.commit()
    conn.close()
    return inserted


# ---------------------------------------------------------------------------
# Recall via graph traversal
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def recall(statement: str, top_k: int = 3) -> list[dict]:
    """
    Recall relevant triples by graph traversal.

    Steps:
    1. Extract key tokens from the statement
    2. Find entities in the graph whose name overlaps with statement tokens
    3. Find triples connected to those entities
    4. Also find triples whose relation matches statement tokens
    5. Score by overlap, return top_k with passage + summary

    Returns list of dicts: filing_id, page_number, passage, section,
                           summary, filing_date, score, subject, relation, value
    """
    global _session_hits

    conn = _get_conn()
    triple_count = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]

    if triple_count == 0:
        conn.close()
        return []

    stmt_tokens = set(_tokenize(statement))

    # --- Step 1: Find matching entities ---
    all_entities = conn.execute("SELECT id, name, type FROM entities").fetchall()
    matching_entity_ids = set()
    for ent in all_entities:
        ent_tokens = set(_tokenize(ent["name"]))
        if ent_tokens & stmt_tokens:
            matching_entity_ids.add(ent["id"])

    # --- Step 2: Find matching relations ---
    all_relations = conn.execute("SELECT DISTINCT relation FROM triples").fetchall()
    matching_relations = []
    for rel in all_relations:
        rel_tokens = set(_tokenize(rel["relation"]))
        if rel_tokens & stmt_tokens:
            matching_relations.append(rel["relation"])

    # --- Step 3: Retrieve candidate triples ---
    candidates = []
    seen_ids: set = set()

    # By entity
    if matching_entity_ids:
        placeholders = ",".join("?" * len(matching_entity_ids))
        rows = conn.execute(
            f"""SELECT t.*, e.name as subject_name, e.type as subject_type
                FROM triples t JOIN entities e ON t.subject_id = e.id
                WHERE t.subject_id IN ({placeholders})
                ORDER BY t.created_at DESC""",
            list(matching_entity_ids),
        ).fetchall()
        for r in rows:
            if r["id"] not in seen_ids:
                candidates.append(r)
                seen_ids.add(r["id"])

    # By relation
    if matching_relations:
        placeholders = ",".join("?" * len(matching_relations))
        rows = conn.execute(
            f"""SELECT t.*, e.name as subject_name, e.type as subject_type
                FROM triples t JOIN entities e ON t.subject_id = e.id
                WHERE t.relation IN ({placeholders})
                ORDER BY t.created_at DESC""",
            matching_relations,
        ).fetchall()
        for r in rows:
            if r["id"] not in seen_ids:
                candidates.append(r)
                seen_ids.add(r["id"])

    # By passage text — find triples whose passage contains statement tokens
    # This catches cases where entity/relation names don't match but the passage does
    all_triples = conn.execute(
        """SELECT t.*, e.name as subject_name, e.type as subject_type
           FROM triples t JOIN entities e ON t.subject_id = e.id"""
    ).fetchall()
    for row in all_triples:
        if row["id"] in seen_ids:
            continue
        passage_tokens = set(_tokenize(row["passage"] + " " + row["summary"]))
        if len(stmt_tokens & passage_tokens) >= 2:
            candidates.append(row)
            seen_ids.add(row["id"])

    conn.close()

    if not candidates:
        return []

    # --- Step 4: Score by token overlap with statement ---
    scored = []
    for row in candidates:
        # Score = overlap of statement tokens with subject + relation + value + summary + passage
        doc_tokens = set(_tokenize(
            f"{row['subject_name']} {row['relation']} {row['object_value']} {row['summary']} {row['passage']}"
        ))
        overlap = len(stmt_tokens & doc_tokens)
        score = overlap / max(len(stmt_tokens), 1)
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by (filing_id, page_number) — take best score per page
    # Minimum score threshold — require at least 3 matching tokens
    min_score = 3.0 / max(len(stmt_tokens), 1)

    seen_pages = set()
    results = []
    for score, row in scored:
        if score < min_score:
            continue
        page_key = (row["filing_id"], row["page_number"])
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        results.append({
            "filing_id":   row["filing_id"],
            "page_number": row["page_number"],
            "passage":     row["passage"],
            "section":     None,
            "summary":     row["summary"],
            "filing_date": row["filing_date"],
            "score":       round(score, 4),
            "subject":     row["subject_name"],
            "relation":    row["relation"],
            "value":       row["object_value"],
            "confidence":  row["confidence"],
        })
        if len(results) >= top_k:
            break

    if results:
        _session_hits += len(results)

    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def clear_memory() -> None:
    """Delete all stored entities and triples."""
    conn = _get_conn()
    conn.execute("DELETE FROM triples")
    conn.execute("DELETE FROM entities")
    conn.commit()
    conn.close()
    print("[memory] Cleared knowledge graph.")


def dump_graph() -> dict:
    """Return the full graph as a dict — useful for debugging."""
    conn = _get_conn()
    entities = [dict(r) for r in conn.execute("SELECT * FROM entities").fetchall()]
    triples  = [dict(r) for r in conn.execute(
        "SELECT t.*, e.name as subject_name FROM triples t JOIN entities e ON t.subject_id = e.id"
    ).fetchall()]
    conn.close()
    return {"entities": entities, "triples": triples}
