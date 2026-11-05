"""
llm.py — Thin wrapper around Ollama's OpenAI-compatible API.

Ollama automatically uses GPU if available — no extra config needed.
All calls are counted and token usage is tracked here.
"""

import json
import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Client setup — points to local Ollama instance
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",  # Ollama doesn't need a real key
)

# ---------------------------------------------------------------------------
# Usage counters
# ---------------------------------------------------------------------------
_usage: dict[str, int] = {
    "llm_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}


def reset_usage() -> None:
    """Reset LLM usage counters to zero."""
    for k in _usage:
        _usage[k] = 0


def get_usage() -> dict[str, int]:
    """Return a copy of current LLM usage counters."""
    return dict(_usage)


# ---------------------------------------------------------------------------
# Core call function
# ---------------------------------------------------------------------------
def call_llm(
    messages: list[dict],
    tools: list[dict] | None = None,  # kept for API compatibility, ignored
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Call the Ollama LLM and return a structured response.

    We use plain text responses (no OpenAI function-calling) because
    local models like qwen2.5:7b are more reliable with JSON-in-text
    than with the OpenAI tools API format.

    The model is instructed to output either:
      {"tool": "name", "args": {...}}   — tool call
      {"verdict": ..., ...}             — final verdict

    Args:
        messages:    List of {"role": ..., "content": ...} dicts
        tools:       Ignored (kept for API compat)
        temperature: Sampling temperature (0 = deterministic)

    Returns:
        {
            "content": str,              # raw text from the model
            "tool_call": {               # parsed if model wants a tool
                "name": str,
                "arguments": dict
            } or None,
            "raw": the full OpenAI response object
        }
    """
    _usage["llm_calls"] += 1

    response = _client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages,
        temperature=temperature,
    )

    # Track token usage
    if response.usage:
        _usage["input_tokens"] += response.usage.prompt_tokens or 0
        _usage["output_tokens"] += response.usage.completion_tokens or 0

    content = response.choices[0].message.content or ""

    # Try to detect a tool call in the response
    tool_call = _parse_tool_call(content)

    return {
        "content": content,
        "tool_call": tool_call,
        "raw": response,
    }


def _parse_tool_call(text: str) -> dict | None:
    """
    Check if the model's text response is a tool call JSON.
    Returns {"name": ..., "arguments": ...} or None.
    """
    data = extract_json(text)
    if data and "tool" in data and "args" in data:
        return {
            "name": data["tool"],
            "arguments": data["args"],
        }
    return None


def extract_json(text: str) -> dict | None:
    """
    Try to extract a JSON object from a text response.
    Returns None if no valid JSON found.
    Handles cases where the model wraps JSON in markdown code blocks.
    """
    if not text:
        return None

    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        text = inner.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object within the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None
