"""
numeric.py — Utilities for normalising financial figures.

SEC filings express figures in many formats:
  "$3,055,555 thousand"  ->  3055555000
  "$3.06 billion"        ->  3060000000
  "963,872"              ->  963872000   (when context says "in thousands")
  "119%"                 ->  1.19

These helpers let the agent compare figures without being tripped up
by unit differences or rounding.
"""

import re
from decimal import Decimal, ROUND_HALF_UP


# ---------------------------------------------------------------------------
# Unit multipliers
# ---------------------------------------------------------------------------
_MULTIPLIERS = {
    "trillion": Decimal("1_000_000_000_000"),
    "billion":  Decimal("1_000_000_000"),
    "million":  Decimal("1_000_000"),
    "thousand": Decimal("1_000"),
    "b":        Decimal("1_000_000_000"),
    "m":        Decimal("1_000_000"),
    "k":        Decimal("1_000"),
}


def parse_financial_figure(text: str) -> Decimal | None:
    """
    Parse a financial figure string into a Decimal number of dollars.

    Examples:
        "$3.06 billion"     -> 3060000000
        "$3,055,555"        -> 3055555          (raw, no unit = as-is)
        "963,872 thousand"  -> 963872000
        "119%"              -> 1.19
        "$4.3 million"      -> 4300000

    Returns None if the text cannot be parsed.
    """
    if not text:
        return None

    text = text.strip().lower()

    # Handle percentages
    pct_match = re.search(r"([\d,\.]+)\s*%", text)
    if pct_match:
        try:
            return Decimal(pct_match.group(1).replace(",", "")) / Decimal("100")
        except Exception:
            return None

    # Remove currency symbols and commas
    text = text.replace("$", "").replace(",", "").strip()

    # Try to match: number + optional unit
    match = re.search(
        r"([\d]+(?:\.[\d]+)?)\s*(trillion|billion|million|thousand|b|m|k)?",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None

    try:
        value = Decimal(match.group(1))
        unit = (match.group(2) or "").lower()
        multiplier = _MULTIPLIERS.get(unit, Decimal("1"))
        return value * multiplier
    except Exception:
        return None


def normalize_to_dollars(value: Decimal | int | float, unit: str = "") -> Decimal:
    """
    Convert a raw number to dollars given its reporting unit.

    Args:
        value: The raw number from the filing
        unit:  "thousands", "millions", "billions", or "" for dollars

    Example:
        normalize_to_dollars(3055555, "thousands") -> 3055555000
    """
    unit = unit.lower().strip()
    multiplier = Decimal("1")
    for key, mult in _MULTIPLIERS.items():
        if key in unit:
            multiplier = mult
            break
    return Decimal(str(value)) * multiplier


def figures_agree(
    a: str,
    b: str,
    tolerance: float = 0.01,
    a_unit: str = "",
    b_unit: str = "",
) -> bool:
    """
    Return True if two financial figure strings refer to the same amount
    within the given relative tolerance (default 1%).

    Examples:
        figures_agree("$3.06 billion", "$3,055,555 thousand") -> True
        figures_agree("963,872 thousand", "$963.9 million")   -> True
        figures_agree("$3.44 billion", "$3.06 billion")       -> False
    """
    val_a = parse_financial_figure(a) if not a_unit else normalize_to_dollars(
        Decimal(a.replace("$", "").replace(",", "").strip()), a_unit
    )
    val_b = parse_financial_figure(b) if not b_unit else normalize_to_dollars(
        Decimal(b.replace("$", "").replace(",", "").strip()), b_unit
    )

    if val_a is None or val_b is None:
        return False
    if val_a == 0 and val_b == 0:
        return True
    if val_a == 0 or val_b == 0:
        return False

    diff = abs(val_a - val_b) / max(abs(val_a), abs(val_b))
    return float(diff) <= tolerance


def extract_figures_from_text(text: str) -> list[dict]:
    """
    Extract all financial figures from a text passage.
    Returns list of {"raw": str, "value": Decimal, "context": str}.
    Useful for the agent to find numbers in a passage.
    """
    results = []
    seen_raw = set()
    # Match patterns like $3.06 billion, $3,055,555, 963,872, 119%, 4,017,540
    patterns = [
        r"\$[\d,]+(?:\.\d+)?\s*(?:trillion|billion|million|thousand|b|m|k)?",
        r"[\d,]+(?:\.\d+)?\s*(?:trillion|billion|million|thousand)\b",
        r"[\d,]+(?:\.\d+)?\s*%",
        r"\b[\d]{1,3}(?:,[\d]{3})+(?:\.\d+)?\b",  # bare comma-separated numbers like 4,017,540
    ]
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            raw = match.group(0).strip()
            if raw in seen_raw:
                continue
            seen_raw.add(raw)
            val = parse_financial_figure(raw)
            if val is not None:
                start = max(0, match.start() - 40)
                end = min(len(text), match.end() + 40)
                ctx = text[start:end].replace("\n", " ").strip()
                results.append({"raw": raw, "value": val, "context": ctx})
    return results
