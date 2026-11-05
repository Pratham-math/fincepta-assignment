"""
tools.py — Central tool registry for the fact-checker agents.

All agent tool calls must go through `run_tool()`.
This gives us:
  - A single place to add new tools (just add to TOOLS dict)
  - Automatic call counting for evaluation
  - Consistent error handling

Usage:
    from src.tools import run_tool, get_tool_schemas, reset_counters, get_counters
"""

from typing import Any
from src.retrieval import list_sections, search_filings, read_passage

# ---------------------------------------------------------------------------
# Tool definitions
# Each entry: function + JSON schema for the LLM
# ---------------------------------------------------------------------------

TOOLS: dict[str, dict] = {
    "list_sections": {
        "fn": list_sections,
        "schema": {
            "type": "function",
            "function": {
                "name": "list_sections",
                "description": (
                    "List the section headings found in a specific filing. "
                    "Use this to discover which sections exist before searching."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filing_id": {
                            "type": "string",
                            "enum": ["10k_fy2024", "10q_jul2024", "10q_oct2024"],
                            "description": "The filing to list sections for.",
                        }
                    },
                    "required": ["filing_id"],
                },
            },
        },
    },
    "search_filings": {
        "fn": search_filings,
        "schema": {
            "type": "function",
            "function": {
                "name": "search_filings",
                "description": (
                    "BM25 keyword search over filing pages. "
                    "Returns the top matching passages with snippet previews. "
                    "Use read_passage to get the full text of a promising result. "
                    "Optionally filter by filing_id or section."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language search query.",
                        },
                        "filing_id": {
                            "type": "string",
                            "enum": ["10k_fy2024", "10q_jul2024", "10q_oct2024"],
                            "description": "Restrict search to one filing (optional).",
                        },
                        "section": {
                            "type": "string",
                            "description": (
                                "Restrict to pages whose section label contains "
                                "this string, e.g. 'Item 3' (optional)."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default 6).",
                            "default": 6,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    },
    "read_passage": {
        "fn": read_passage,
        "schema": {
            "type": "function",
            "function": {
                "name": "read_passage",
                "description": (
                    "Fetch the full text of a specific page by its passage_id "
                    "(format: 'filing_id:page_number', e.g. '10k_fy2024:68'). "
                    "Use include_neighbors=true to also fetch adjacent pages "
                    "when a table or paragraph spans multiple pages."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "passage_id": {
                            "type": "string",
                            "description": "passage_id from search results, e.g. '10k_fy2024:68'.",
                        },
                        "include_neighbors": {
                            "type": "boolean",
                            "description": "Also return the previous and next pages (default false).",
                            "default": False,
                        },
                    },
                    "required": ["passage_id"],
                },
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Call counters — reset between runs
# ---------------------------------------------------------------------------
_counters: dict[str, int] = {"total": 0}
for _name in TOOLS:
    _counters[_name] = 0


def reset_counters() -> None:
    """Reset all tool call counters to zero."""
    for key in _counters:
        _counters[key] = 0


def get_counters() -> dict[str, int]:
    """Return a copy of the current tool call counts."""
    return dict(_counters)


# ---------------------------------------------------------------------------
# Central dispatch
# ---------------------------------------------------------------------------
def run_tool(name: str, args: dict[str, Any]) -> Any:
    """
    Execute a tool by name with the given arguments.

    Returns the tool's result dict, or an error dict if the tool is unknown
    or raises an exception.
    """
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'. Available: {list(TOOLS.keys())}"}

    # Increment counters
    _counters["total"] += 1
    _counters[name] = _counters.get(name, 0) + 1

    try:
        result = TOOLS[name]["fn"](**args)
        return result
    except TypeError as e:
        return {"error": f"Bad arguments for tool '{name}': {e}"}
    except Exception as e:
        return {"error": f"Tool '{name}' raised an error: {e}"}


def get_tool_schemas() -> list[dict]:
    """Return the list of tool schemas for passing to the LLM."""
    return [t["schema"] for t in TOOLS.values()]
