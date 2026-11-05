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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Build provider list — priority: OpenAI → Gemini → OpenRouter → Ollama
_providers = []
if OPENAI_API_KEY:
    _providers.append({
        "name": "openai",
        "client": OpenAI(api_key=OPENAI_API_KEY),
        "model": OPENAI_MODEL,
        "headers": None,
        "exhausted": False,
    })
if GEMINI_API_KEY:
    _providers.append({
        "name": "gemini",
        "client": OpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY),
        "model": GEMINI_MODEL,
        "headers": None,
        "exhausted": False,
    })
if OPENROUTER_API_KEY:
    _providers.append({
        "name": "openrouter",
        "client": OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY),
        "model": OPENROUTER_MODEL,
        "headers": {
            "HTTP-Referer": "https://github.com/Pratham-math/fincepta-assignment",
            "X-Title": "Filing Fact-Checker",
        },
        "exhausted": False,
    })
if not _providers:
    _providers.append({
        "name": "ollama",
        "client": OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama"),
        "model": OLLAMA_MODEL,
        "headers": None,
        "exhausted": False,
    })

_current_idx = 0  # index into _providers

def _current_provider():
    return _providers[_current_idx]

print(f"[llm] Providers: {[p['name'] + ':' + p['model'] for p in _providers]}")
print(f"[llm] Active: {_current_provider()['name']}:{_current_provider()['model']}")

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


def _switch_provider() -> bool:
    """Switch to next available provider. Returns True if switched."""
    global _current_idx
    for i in range(len(_providers)):
        if i != _current_idx and not _providers[i]["exhausted"]:
            _current_idx = i
            p = _providers[i]
            print(f"[llm] Switched to {p['name']}:{p['model']}")
            return True
    return False


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

    import time as _time

    response = None
    last_error = None

    # Try each provider until one works
    for attempt in range(4):
        provider = _current_provider()

        if provider["exhausted"]:
            # Move to next provider
            _switch_provider()
            continue

        try:
            kwargs = {
                "model": provider["model"],
                "messages": messages,
                "temperature": temperature,
            }
            if provider.get("headers"):
                kwargs["extra_headers"] = provider["headers"]

            response = provider["client"].chat.completions.create(**kwargs)
            break

        except Exception as e:
            err_str = str(e)
            # Check for quota exhaustion (not just temporary rate limit)
            if "quota" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str:
                print(f"  [llm] {provider['name']} quota exhausted, switching provider...")
                provider["exhausted"] = True
                _switch_provider()
                continue
            # Temporary rate limit / 503 → retry with backoff
            if any(code in err_str for code in ["429", "503"]) and attempt < 3:
                # If next provider exists, switch; otherwise wait
                if _switch_provider():
                    continue
                wait = 2 ** (attempt + 1)
                print(f"  [llm] {provider['name']} rate limited, waiting {wait}s...")
                _time.sleep(wait)
                continue
            last_error = e
            raise

    if response is None:
        raise last_error or RuntimeError("All providers exhausted")

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
