"""
Core data schemas for the Filing Fact-Checker.
All verdicts and evidence must conform to these models.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    OUTDATED = "OUTDATED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class FinishReason(str, Enum):
    CONFIDENT = "confident"
    STEP_LIMIT = "step_limit"
    ERROR = "error"


class Evidence(BaseModel):
    """A single passage used as evidence for a verdict."""
    filing_id: str = Field(description="e.g. '10k_fy2024', '10q_jul2024', '10q_oct2024'")
    page_number: int = Field(description="0-indexed PDF page number")
    passage: str = Field(description="Exact quoted text from the filing")
    section: Optional[str] = Field(default=None, description="Section label if known, e.g. 'Item 1A'")


class Usage(BaseModel):
    """Tracks LLM and tool call counts for a single check."""
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    memory_hits: int = 0


class VerificationResult(BaseModel):
    """The full result returned for a single statement check."""
    statement: str
    verdict: Verdict
    explanation: str
    evidence: list[Evidence] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    finish_reason: FinishReason = FinishReason.CONFIDENT


class FilingPage(BaseModel):
    """A single extracted page from a filing, stored in the index."""
    filing_id: str
    period: str           # e.g. "FY2024", "Q2FY2025", "Q3FY2025"
    filing_date: str      # ISO date string, e.g. "2024-03-07"
    page_number: int      # 0-indexed
    text: str             # cleaned text for search
    raw_text: str         # original extracted text for evidence quotation
    is_toc: bool = False  # True if this looks like a table of contents page
    section: Optional[str] = None  # section label if detected
