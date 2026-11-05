"""
ingest.py — Extract text from CrowdStrike SEC filings and build data/index.json

Usage:
    python ingest.py

Reads PDFs from the project root, writes data/index.json.
Each entry is one page with metadata for BM25 search and evidence citation.
"""

import json
import re
import sys
from pathlib import Path

import pdfplumber

# ---------------------------------------------------------------------------
# Filing manifest — edit here if filenames change
# ---------------------------------------------------------------------------
FILINGS = [
    {
        "filing_id": "10k_fy2024",
        "filename": "crwd-20240131.pdf",
        "period": "FY2024",
        "filing_date": "2024-03-07",   # approximate SEC filing date
        "filing_type": "10-K",
    },
    {
        "filing_id": "10q_jul2024",
        "filename": "0001535527-24-000020.pdf",
        "period": "Q2FY2025",
        "filing_date": "2024-09-05",
        "filing_type": "10-Q",
    },
    {
        "filing_id": "10q_oct2024",
        "filename": "crwd-20241031.pdf",
        "period": "Q3FY2025",
        "filing_date": "2024-12-05",
        "filing_type": "10-Q",
    },
]

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
INDEX_FILE = DATA_DIR / "index.json"

# ---------------------------------------------------------------------------
# Section heading patterns for SEC filings
# ---------------------------------------------------------------------------
SECTION_PATTERNS = [
    (r"PART\s+[IVX]+", "Part"),
    (r"Item\s+\d+[A-Z]?\.", "Item"),
    (r"ITEM\s+\d+[A-Z]?\.", "Item"),
]

# Patterns that indicate a table of contents page
# Note: "Table of Contents" appears as a nav header on EVERY page in these PDFs,
# so we only flag actual TOC pages by looking for the numbered item listing pattern.
TOC_PATTERNS = [
    r"TABLE\s+OF\s+CONTENTS\s*\nPage",   # heading followed by "Page No."
    r"TABLE\s+OF\s+CONTENTS\s*\nPart",   # heading followed by "Part I"
    r"INDEX\s+TO\s+FINANCIAL\s+STATEMENTS",
]

# Headers/footers to strip from search text (but keep in raw_text)
HEADER_FOOTER_PATTERNS = [
    r"\d+/\d+/\d+,\s+\d+:\d+\s+[AP]M\s+crwd-\S+",   # browser timestamp header
    r"https?://\S+",                                    # URLs
    r"Table of Contents\s*\n",                          # nav header (appears on every page)
    r"^\s*\d+\s*$",                                     # lone page numbers
]


def extract_table_text(table) -> str:
    """Convert a pdfplumber table to readable text preserving row structure."""
    if not table:
        return ""
    lines = []
    for row in table:
        # Filter None cells, join with tab
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        line = " | ".join(c for c in cells if c)
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


def detect_section(text: str) -> str | None:
    """Return the first section heading found in the page text."""
    for pattern, label in SECTION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Grab a bit more context after the match for the full heading
            start = match.start()
            snippet = text[start:start + 60].strip().split("\n")[0]
            return snippet[:60]
    return None


def is_toc_page(text: str) -> bool:
    """Return True if this page IS the actual table of contents.
    
    These PDFs have 'Table of Contents' as a nav header on every page,
    so we require the specific pattern of a TOC heading followed by
    numbered item listings with page references.
    """
    for pattern in TOC_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def clean_for_search(text: str) -> str:
    """Strip headers, footers and excess whitespace for cleaner BM25 indexing."""
    cleaned = text
    for pattern in HEADER_FOOTER_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.MULTILINE | re.IGNORECASE)
    # Collapse multiple spaces/newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def extract_filing(filing_meta: dict) -> list[dict]:
    """Extract all pages from one PDF, returning a list of FilingPage dicts."""
    pdf_path = ROOT / filing_meta["filename"]
    if not pdf_path.exists():
        print(f"  ERROR: {pdf_path} not found — skipping")
        return []

    pages = []
    print(f"  Opening {pdf_path.name} ...")

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"  {total} pages found")

        for i, page in enumerate(pdf.pages):
            # Extract plain text
            raw_text = page.extract_text() or ""

            # Extract tables and append their text
            tables_text = ""
            try:
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        tables_text += extract_table_text(table) + "\n"
            except Exception:
                pass  # table extraction is best-effort

            # Combine: raw text first, then any table content not already there
            combined_raw = raw_text
            if tables_text.strip():
                combined_raw = combined_raw + "\n\n[TABLE]\n" + tables_text

            # Skip completely empty pages
            if not combined_raw.strip():
                continue

            cleaned = clean_for_search(combined_raw)
            section = detect_section(combined_raw)
            toc = is_toc_page(combined_raw)

            page_entry = {
                "filing_id": filing_meta["filing_id"],
                "period": filing_meta["period"],
                "filing_date": filing_meta["filing_date"],
                "filing_type": filing_meta["filing_type"],
                "page_number": i,          # 0-indexed PDF page
                "page_label": str(i + 1),  # human-readable label
                "text": cleaned,           # for BM25 search
                "raw_text": combined_raw,  # for evidence quotation
                "is_toc": toc,
                "section": section,
            }
            pages.append(page_entry)

            if (i + 1) % 50 == 0:
                print(f"    ... {i + 1}/{total} pages done")

    print(f"  Extracted {len(pages)} non-empty pages from {filing_meta['filename']}")
    return pages


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_pages = []
    for filing in FILINGS:
        print(f"\nProcessing {filing['filing_id']} ({filing['filing_type']}, {filing['period']}) ...")
        pages = extract_filing(filing)
        all_pages.extend(pages)

    print(f"\nTotal pages extracted: {len(all_pages)}")
    print(f"Writing index to {INDEX_FILE} ...")

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)

    print("Done.")

    # Quick sanity summary
    from collections import Counter
    counts = Counter(p["filing_id"] for p in all_pages)
    toc_counts = Counter(p["filing_id"] for p in all_pages if p["is_toc"])
    print("\nSummary:")
    for filing_id, count in counts.items():
        print(f"  {filing_id}: {count} pages ({toc_counts.get(filing_id, 0)} TOC pages)")


if __name__ == "__main__":
    main()
