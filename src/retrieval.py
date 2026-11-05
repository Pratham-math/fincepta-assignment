"""
retrieval.py — BM25-based search over the extracted filing pages.

Provides three functions used as agent tools:
  - list_sections(filing_id)
  - search_filings(query, filing_id, section, top_k)
  - read_passage(passage_id, include_neighbors)
"""

import json
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Load index once at import time
# ---------------------------------------------------------------------------
_INDEX_FILE = Path(__file__).parent.parent / "data" / "index.json"

def _load_index() -> list[dict]:
    with open(_INDEX_FILE, encoding="utf-8") as f:
        return json.load(f)

_PAGES: list[dict] = _load_index()

# Build a passage_id -> page lookup  (filing_id:page_number)
_ID_MAP: dict[str, dict] = {
    f"{p['filing_id']}:{p['page_number']}": p for p in _PAGES
}

# BM25 index — tokenise on whitespace/punctuation
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

_CORPUS_TOKENS = [_tokenize(p["text"]) for p in _PAGES]
_BM25 = BM25Okapi(_CORPUS_TOKENS)

# Valid filing IDs
VALID_FILING_IDS = {"10k_fy2024", "10q_jul2024", "10q_oct2024"}

# Filing date order for temporal reasoning (oldest first)
FILING_ORDER = ["10k_fy2024", "10q_jul2024", "10q_oct2024"]


# ---------------------------------------------------------------------------
# Tool 1 — list_sections
# ---------------------------------------------------------------------------
def list_sections(filing_id: str) -> dict:
    """
    Return a list of detected section headings for a given filing.

    Args:
        filing_id: One of '10k_fy2024', '10q_jul2024', '10q_oct2024'

    Returns:
        {"filing_id": ..., "sections": [{"section": ..., "page_number": ...}, ...]}
    """
    if filing_id not in VALID_FILING_IDS:
        return {"error": f"Unknown filing_id '{filing_id}'. Valid: {sorted(VALID_FILING_IDS)}"}

    sections = []
    seen = set()
    for p in _PAGES:
        if p["filing_id"] != filing_id:
            continue
        s = p.get("section")
        if s and s not in seen:
            seen.add(s)
            sections.append({"section": s, "page_number": p["page_number"]})

    return {"filing_id": filing_id, "sections": sections}


# ---------------------------------------------------------------------------
# Tool 2 — search_filings
# ---------------------------------------------------------------------------
def search_filings(
    query: str,
    filing_id: Optional[str] = None,
    section: Optional[str] = None,
    top_k: int = 6,
) -> dict:
    """
    BM25 search over filing pages.  Returns the top matching passages.

    Args:
        query:     Natural-language search query
        filing_id: Restrict to one filing (optional)
        section:   Restrict to pages whose section label contains this string (optional)
        top_k:     Number of results to return (default 6)

    Returns:
        {"query": ..., "results": [{"passage_id", "filing_id", "period",
         "filing_date", "page_number", "section", "snippet", "score"}, ...]}
    """
    if not query or not query.strip():
        return {"error": "query must not be empty"}

    # Filter candidate pages
    candidates = _PAGES
    if filing_id:
        if filing_id not in VALID_FILING_IDS:
            return {"error": f"Unknown filing_id '{filing_id}'"}
        candidates = [p for p in candidates if p["filing_id"] == filing_id]
    if section:
        section_lower = section.lower()
        candidates = [
            p for p in candidates
            if p.get("section") and section_lower in p["section"].lower()
        ]

    if not candidates:
        return {"query": query, "results": []}

    # BM25 score against the filtered subset
    query_tokens = _tokenize(query)
    # Build a mini BM25 over the filtered set
    mini_tokens = [_tokenize(p["text"]) for p in candidates]
    mini_bm25 = BM25Okapi(mini_tokens)
    scores = mini_bm25.get_scores(query_tokens)

    # Sort by score descending, take top_k
    ranked = sorted(
        zip(scores, candidates), key=lambda x: x[0], reverse=True
    )[:top_k]

    results = []
    for score, page in ranked:
        if score <= 0:
            continue
        passage_id = f"{page['filing_id']}:{page['page_number']}"
        # Return a snippet (first 400 chars of cleaned text)
        snippet = page["text"][:400].strip()
        results.append({
            "passage_id": passage_id,
            "filing_id": page["filing_id"],
            "period": page["period"],
            "filing_date": page["filing_date"],
            "page_number": page["page_number"],
            "section": page.get("section"),
            "snippet": snippet,
            "score": round(float(score), 4),
        })

    return {"query": query, "results": results}


# ---------------------------------------------------------------------------
# Tool 3 — read_passage
# ---------------------------------------------------------------------------
def read_passage(passage_id: str, include_neighbors: bool = False) -> dict:
    """
    Return the full text of a specific page by its passage_id.

    Args:
        passage_id:        Format 'filing_id:page_number', e.g. '10k_fy2024:68'
        include_neighbors: If True, also return the previous and next pages

    Returns:
        {"passage_id": ..., "filing_id": ..., "page_number": ...,
         "section": ..., "filing_date": ..., "text": ...,
         "prev": {...} or null, "next": {...} or null}
    """
    if passage_id not in _ID_MAP:
        return {"error": f"passage_id '{passage_id}' not found. Format: 'filing_id:page_number'"}

    page = _ID_MAP[passage_id]

    def _page_summary(p: dict) -> dict:
        return {
            "passage_id": f"{p['filing_id']}:{p['page_number']}",
            "filing_id": p["filing_id"],
            "period": p["period"],
            "filing_date": p["filing_date"],
            "page_number": p["page_number"],
            "section": p.get("section"),
            "text": p["raw_text"],   # full original text for evidence quotation
        }

    result = _page_summary(page)
    result["prev"] = None
    result["next"] = None

    if include_neighbors:
        fid = page["filing_id"]
        pnum = page["page_number"]
        prev_id = f"{fid}:{pnum - 1}"
        next_id = f"{fid}:{pnum + 1}"
        if prev_id in _ID_MAP:
            result["prev"] = _page_summary(_ID_MAP[prev_id])
        if next_id in _ID_MAP:
            result["next"] = _page_summary(_ID_MAP[next_id])

    return result
