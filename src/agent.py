"""
agent.py — LangGraph fact-checker graph.

Graph structure:
    executor_node  →  tool_node  →  executor_node  (loop)
         ↓ (verdict or step limit)
        END

Phases:
  - Phase 3: Executor only (no Decomposer, no Memory)
  - Phase 4 will add: decomposer_node before executor, memory recall/write
"""

import json
from typing import Annotated, Any, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.prompts import EXECUTOR_SYSTEM_PROMPT, FEW_SHOT_MESSAGES
from src.llm import call_llm, extract_json, get_usage, reset_usage
from src.tools import run_tool, get_counters, reset_counters
from src.schemas import Verdict, FinishReason, Evidence, Usage, VerificationResult

MAX_STEPS = int(__import__("os").getenv("MAX_EXECUTOR_STEPS", "5"))


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    statement: str                          # the claim to verify
    messages: Annotated[list, add_messages] # full conversation history
    steps: int                              # executor turns used
    verdict_data: dict | None               # set when executor returns a verdict
    finish_reason: str                      # "confident" | "step_limit" | "error"
    error: str | None                       # last error message if any


# ---------------------------------------------------------------------------
# Executor Node
# ---------------------------------------------------------------------------
def executor_node(state: AgentState) -> dict:
    """
    Call the LLM. It will either:
      (a) return a tool call JSON: {"tool": "name", "args": {...}}
      (b) return a verdict JSON: {"verdict": ..., "explanation": ..., "evidence": [...]}
    Handles malformed output with one repair attempt.
    """
    messages = state["messages"]
    steps = state["steps"] + 1

    # Hard stop
    if steps > MAX_STEPS:
        return {
            "steps": steps,
            "verdict_data": {
                "verdict": Verdict.NOT_ENOUGH_INFO,
                "explanation": "Step limit reached without sufficient evidence.",
                "evidence": [],
            },
            "finish_reason": FinishReason.STEP_LIMIT,
        }

    # Convert LangGraph message objects to plain dicts for our LLM wrapper
    plain_messages = _to_plain_messages(messages)

    response = call_llm(messages=plain_messages, temperature=0.0)
    content = response["content"] or ""

    # --- Model wants to call a tool ---
    if response["tool_call"]:
        tc = response["tool_call"]
        assistant_msg = {"role": "assistant", "content": content}
        sentinel_msg = {
            "role": "system",
            "content": f"__TOOL_CALL__:{json.dumps(tc)}",
        }
        return {
            "messages": [assistant_msg, sentinel_msg],
            "steps": steps,
            "verdict_data": None,
            "finish_reason": FinishReason.CONFIDENT,
            "error": None,
        }

    # --- Model returned text — try to parse a verdict ---
    verdict_data = extract_json(content)

    if verdict_data and _is_valid_verdict(verdict_data):
        return {
            "messages": [{"role": "assistant", "content": content}],
            "steps": steps,
            "verdict_data": verdict_data,
            "finish_reason": FinishReason.CONFIDENT,
            "error": None,
        }

    # --- Malformed output — one repair attempt ---
    repair_prompt = (
        "Your previous response was not valid JSON or was missing required fields. "
        "Please respond with ONLY valid JSON. Either a tool call:\n"
        '{"tool": "search_filings", "args": {"query": "..."}}\n'
        "Or a verdict:\n"
        '{"verdict": "SUPPORTED", "explanation": "...", "evidence": [...]}\n'
        "verdict must be one of: SUPPORTED, CONTRADICTED, OUTDATED, NOT_ENOUGH_INFO"
    )
    repair_messages = plain_messages + [
        {"role": "assistant", "content": content},
        {"role": "user", "content": repair_prompt},
    ]
    repair_response = call_llm(messages=repair_messages, temperature=0.0)
    repair_content = repair_response["content"] or ""

    # Check if repair gave a tool call
    if repair_response["tool_call"]:
        tc = repair_response["tool_call"]
        sentinel_msg = {
            "role": "system",
            "content": f"__TOOL_CALL__:{json.dumps(tc)}",
        }
        return {
            "messages": [
                {"role": "assistant", "content": content},
                {"role": "user", "content": repair_prompt},
                {"role": "assistant", "content": repair_content},
                sentinel_msg,
            ],
            "steps": steps,
            "verdict_data": None,
            "finish_reason": FinishReason.CONFIDENT,
            "error": None,
        }

    verdict_data = extract_json(repair_content)
    if verdict_data and _is_valid_verdict(verdict_data):
        return {
            "messages": [
                {"role": "assistant", "content": content},
                {"role": "user", "content": repair_prompt},
                {"role": "assistant", "content": repair_content},
            ],
            "steps": steps,
            "verdict_data": verdict_data,
            "finish_reason": FinishReason.CONFIDENT,
            "error": None,
        }

    # Both attempts failed
    return {
        "messages": [{"role": "assistant", "content": content}],
        "steps": steps,
        "verdict_data": {
            "verdict": Verdict.NOT_ENOUGH_INFO,
            "explanation": "Could not produce a structured verdict after two attempts.",
            "evidence": [],
        },
        "finish_reason": FinishReason.ERROR,
        "error": f"Malformed LLM output: {content[:200]}",
    }


def _to_plain_messages(messages: list) -> list[dict]:
    """
    Convert LangGraph message objects (or dicts) to plain dicts
    compatible with the OpenAI API. Filters out __TOOL_CALL__ sentinels.
    """
    plain = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "") or ""
        else:
            # LangGraph wraps messages in objects with .type and .content
            type_map = {
                "human": "user",
                "ai": "assistant",
                "system": "system",
                "tool": "user",
            }
            role = type_map.get(getattr(m, "type", "human"), "user")
            content = getattr(m, "content", "") or ""

        # Skip sentinel messages — they're internal bookkeeping
        if content.startswith("__TOOL_CALL__:"):
            continue

        plain.append({"role": role, "content": content})
    return plain


# ---------------------------------------------------------------------------
# Tool Node
# ---------------------------------------------------------------------------
def tool_node(state: AgentState) -> dict:
    """
    Execute the tool encoded in the last __TOOL_CALL__ sentinel message.
    Removes the sentinel and appends the tool result as a user message.
    """
    messages = state["messages"]

    # Find the last sentinel message (works on both dicts and LangGraph objects)
    tc = None
    for msg in reversed(messages):
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if content and isinstance(content, str) and content.startswith("__TOOL_CALL__:"):
            try:
                tc = json.loads(content[len("__TOOL_CALL__:"):])
            except json.JSONDecodeError:
                pass
            break

    if not tc:
        return {
            "messages": [{"role": "user", "content": "Tool error: no pending tool call found."}],
        }

    tool_name = tc.get("name", "")
    tool_args = tc.get("arguments", {})

    result = run_tool(tool_name, tool_args)
    result_str = json.dumps(result, ensure_ascii=False)

    # Truncate very long results to avoid bloating context
    if len(result_str) > 6000:
        result_str = result_str[:6000] + "\n... [truncated]"

    return {
        "messages": [
            {
                "role": "user",
                "content": f"Tool result for {tool_name}:\n{result_str}",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Router — decides next node after executor
# ---------------------------------------------------------------------------
def route_after_executor(state: AgentState) -> str:
    """Return the name of the next node to visit."""
    # Verdict reached
    if state.get("verdict_data"):
        return END

    # Step limit hit
    if state["steps"] >= MAX_STEPS:
        return END

    # Check for a pending tool call sentinel in messages
    messages = state["messages"]
    for msg in reversed(messages):
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if content and content.startswith("__TOOL_CALL__:"):
            return "tool_node"

    return END


# ---------------------------------------------------------------------------
# Helper — validate verdict structure
# ---------------------------------------------------------------------------
def _is_valid_verdict(data: dict) -> bool:
    """Return True if data has the required verdict fields."""
    if not isinstance(data, dict):
        return False
    if "verdict" not in data:
        return False
    valid_verdicts = {"SUPPORTED", "CONTRADICTED", "OUTDATED", "NOT_ENOUGH_INFO"}
    if data["verdict"] not in valid_verdicts:
        return False
    if "explanation" not in data:
        return False
    if "evidence" not in data:
        return False
    return True


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def _build_graph() -> Any:
    builder = StateGraph(AgentState)

    builder.add_node("executor_node", executor_node)
    builder.add_node("tool_node", tool_node)

    builder.set_entry_point("executor_node")

    builder.add_conditional_edges(
        "executor_node",
        route_after_executor,
        {
            "tool_node": "tool_node",
            END: END,
        },
    )
    builder.add_edge("tool_node", "executor_node")

    return builder.compile()


_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_check(
    statement: str,
    use_decomposer: bool = False,   # Phase 4 will wire this up
    use_memory: bool = False,        # Phase 5 will wire this up
) -> VerificationResult:
    """
    Check a statement against the filings.

    Args:
        statement:      The claim to verify
        use_decomposer: Whether to run the Decomposer first (Phase 4+)
        use_memory:     Whether to use evidence memory (Phase 5+)

    Returns:
        VerificationResult with verdict, evidence, explanation and usage stats
    """
    reset_usage()
    reset_counters()

    # Initial messages — system prompt + few-shot example + actual task
    initial_messages = [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        *FEW_SHOT_MESSAGES,
        {
            "role": "user",
            "content": (
                f"Verify this statement against the CrowdStrike SEC filings:\n\n"
                f"{statement}\n\n"
                f"Start by calling search_filings."
            ),
        },
    ]

    initial_state: AgentState = {
        "statement": statement,
        "messages": initial_messages,
        "steps": 0,
        "verdict_data": None,
        "finish_reason": FinishReason.CONFIDENT,
        "error": None,
    }

    final_state = _GRAPH.invoke(initial_state)

    # Build result
    verdict_data = final_state.get("verdict_data") or {
        "verdict": Verdict.NOT_ENOUGH_INFO,
        "explanation": "No verdict was reached.",
        "evidence": [],
    }

    # Validate and normalize verdict
    verdict_str = verdict_data.get("verdict", "NOT_ENOUGH_INFO")
    try:
        verdict = Verdict(verdict_str)
    except ValueError:
        verdict = Verdict.NOT_ENOUGH_INFO

    # Build evidence list — validate passage_id if present
    evidence_list = []
    for ev in verdict_data.get("evidence", []):
        if not isinstance(ev, dict):
            continue
        evidence_list.append(
            Evidence(
                filing_id=ev.get("filing_id", "unknown"),
                page_number=int(ev.get("page_number", 0)),
                passage=ev.get("passage", ""),
                section=ev.get("section"),
            )
        )

    # Collect usage
    llm_usage = get_usage()
    tool_usage = get_counters()

    usage = Usage(
        llm_calls=llm_usage["llm_calls"],
        input_tokens=llm_usage["input_tokens"],
        output_tokens=llm_usage["output_tokens"],
        tool_calls=tool_usage["total"],
        memory_hits=0,  # Phase 5
    )

    finish_reason = final_state.get("finish_reason", FinishReason.CONFIDENT)
    try:
        finish_reason = FinishReason(finish_reason)
    except ValueError:
        finish_reason = FinishReason.CONFIDENT

    return VerificationResult(
        statement=statement,
        verdict=verdict,
        explanation=verdict_data.get("explanation", ""),
        evidence=evidence_list,
        usage=usage,
        finish_reason=finish_reason,
    )


def run_check_stream(statement: str):
    """
    Generator that yields step events as the agent works.
    Used by the FastAPI streaming endpoint.

    Yields dicts:
      {"type": "step", "step": N, "action": "tool_call"|"verdict", "detail": ...}
      {"type": "result", "data": VerificationResult.model_dump()}
    """
    reset_usage()
    reset_counters()

    initial_messages = [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        *FEW_SHOT_MESSAGES,
        {
            "role": "user",
            "content": (
                f"Verify this statement against the CrowdStrike SEC filings:\n\n"
                f"{statement}\n\n"
                f"Start by calling search_filings."
            ),
        },
    ]

    initial_state: AgentState = {
        "statement": statement,
        "messages": initial_messages,
        "steps": 0,
        "verdict_data": None,
        "finish_reason": FinishReason.CONFIDENT,
        "error": None,
    }

    # Stream graph updates
    for event in _GRAPH.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "executor_node":
                if node_output.get("verdict_data"):
                    yield {
                        "type": "step",
                        "step": node_output.get("steps", 0),
                        "action": "verdict",
                        "detail": node_output["verdict_data"].get("verdict"),
                    }
                else:
                    # Tool call coming
                    msgs = node_output.get("messages", [])
                    tool_name = None
                    for m in msgs:
                        if isinstance(m, dict) and m.get("tool_calls"):
                            tool_name = m["tool_calls"][0]["function"]["name"]
                    yield {
                        "type": "step",
                        "step": node_output.get("steps", 0),
                        "action": "tool_call",
                        "detail": tool_name,
                    }
            elif node_name == "tool_node":
                msgs = node_output.get("messages", [])
                yield {
                    "type": "step",
                    "step": 0,
                    "action": "tool_result",
                    "detail": f"tool returned {len(msgs)} message(s)",
                }

    # Final result — re-run to get structured output
    # (stream gives us updates but not the final composed state easily)
    result = run_check(statement)
    yield {"type": "result", "data": result.model_dump()}
