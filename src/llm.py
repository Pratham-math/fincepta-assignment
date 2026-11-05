"""
llm.py — LLM wrapper supporting OpenRouter or local Ollama.

Priority:
  1. If OPENROUTER_API_KEY is set → use OpenRouter
  2. Otherwise → use local Ollama at localhost:11434

All calls are counted and token usage is tracked here.
"""

import json
import os
import re
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

if OPENROUTER_API_KEY:
    _client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )
    _model = OPENROUTER_MODEL
    _provider = "openrouter"
else:
    _client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
    )
    _model = OLLAMA_MODEL
    _provider = "ollama"

print(f"[llm] Using {_provider}: {_model}")

# ---------------------------------------------------------------------------
# Usage counters
# ---------------------------------------------------------------------------
_usage: dict[str, int] = {
    "llm_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}


def reset_usage() -> None:
    for k in _usage:
        _usage[k] = 0


def get_usage() -> dict[str, int]:
    return dict(_usage)


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------
def call_llm(
    messages: list[dict],
    tools: list[dict] | None = None,   # kept for API compat, ignored
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Call the LLM and return a structured response.

    We use plain text responses (JSON-in-text) because it works reliably
    with both local and cloud models without requiring native tool-calling.

    Returns:
        {
            "content": str,
            "tool_call": {"name": str, "arguments": dict} | None,
            "raw": response object
        }
    """
    _usage["llm_calls"] += 1

    # OpenRouter requires these headers for ranking/attribution
    extra_headers = {}
    if _provider == "openrouter":
        extra_headers = {
            "HTTP-Referer": "https://github.com/Pratham-math/fincepta-assignment",
            "X-Title": "Filing Fact-Checker",
        }

    # Retry on rate limit (429) with exponential backoff
    import time as _time
    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model=_model,
                messages=messages,
                temperature=temperature,
                extra_headers=extra_headers if extra_headers else None,
            )
            break
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                print(f"  [llm] Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                _time.sleep(wait)
            else:
                raise

    if response.usage:
        _usage["input_tokens"]  += response.usage.prompt_tokens or 0
        _usage["output_tokens"] += response.usage.completion_tokens or 0

    content = response.choices[0].message.content or ""

    # Strip <think>...</think> blocks that some models emit before their JSON
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    tool_call = _parse_tool_call(content)

    return {
        "content": content,
        "tool_call": tool_call,
        "raw": response,
    }


def _parse_tool_call(text: str) -> dict | None:
    """Detect a tool call JSON in the model's text response."""
    data = extract_json(text)
    if data and "tool" in data and "args" in data:
        return {"name": data["tool"], "arguments": data["args"]}
    return None


def extract_json(text: str) -> dict | None:
    """
    Extract a JSON object from text. Handles markdown code fences.
    Returns None if no valid JSON found.
    """
    if not text:
        return None

    text = text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end]).strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find outermost JSON object
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None
