"""
agent.py — LangGraph fact-checker graph.

Graph structure (with decomposer):
    decomposer_node  →  executor_node  →  tool_node  →  executor_node  (loop)
                              ↓ (verdict or step limit)
                             END

use_decomposer=False skips decomposer_node (single-agent baseline).
use_memory=False skips memory recall/write (Phase 5).
"""

import json
import os
from typing import Annotated, Any, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.prompts import EXECUTOR_SYSTEM_PROMPT, DECOMPOSER_SYSTEM_PROMPT
from src.llm import call_llm, extract_json, get_usage, reset_usage
from src.tools import run_tool, get_counters, reset_counters
from src.schemas import Verdict, FinishReason, Evidence, Usage, VerificationResult
from src.numeric import figures_agree, extract_figures_from_text
MAX_STEPS = int(os.getenv("MAX_EXECUTOR_STEPS", "7"))


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    statement: str                          # the original claim
    decomposition: dict | None              # output of decomposer (topic, search_terms, etc.)
    messages: Annotated[list, add_messages] # full conversation history
    steps: int                              # executor turns used
    verdict_data: dict | None               # set when executor returns a verdict
    finish_reason: str
    error: str | None


# ---------------------------------------------------------------------------
# Decomposer Node
# ---------------------------------------------------------------------------
def decomposer_node(state: AgentState) -> dict:
    """
    One LLM call that extracts the key components of the statement:
    topic, time_scope, metric_or_entity, search_terms, relevant_filings.

    On failure (malformed JSON or missing fields), falls back to the
    original statement as a single search term.
    """
    statement = state["statement"]

    messages = [
        {"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Decompose this statement:\n\n{statement}"},
    ]

    response = call_llm(messages=messages, temperature=0.0)
    content = response["content"] or ""
    data = extract_json(content)

    # Validate — must have at minimum search_terms
    if data and isinstance(data.get("search_terms"), list) and len(data["search_terms"]) > 0:
        decomposition = data
    else:
        # One repair attempt
        repair_messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": (
                "Please respond with ONLY the JSON object. "
                "Required fields: topic, time_scope, metric_or_entity, "
                "search_terms (list of strings), relevant_filings (list)."
            )},
        ]
        repair_response = call_llm(messages=repair_messages, temperature=0.0)
        repair_data = extract_json(repair_response["content"] or "")

        if repair_data and isinstance(repair_data.get("search_terms"), list):
            decomposition = repair_data
        else:
            # Fallback — use the raw statement as a single search term
            decomposition = {
                "topic": statement[:80],
                "time_scope": None,
                "metric_or_entity": None,
                "search_terms": [statement[:120]],
                "relevant_filings": ["10k_fy2024", "10q_jul2024", "10q_oct2024"],
                "_fallback": True,
            }

    return {"decomposition": decomposition}


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

    # Hard stop — step limit exceeded
    if steps > MAX_STEPS:
        return {
            "steps": steps,
            "verdict_data": {
                "verdict": Verdict.NOT_ENOUGH_INFO,
                "explanation": f"Step limit ({MAX_STEPS}) reached without sufficient evidence.",
                "evidence": [],
            },
            "finish_reason": FinishReason.STEP_LIMIT,
            "error": "step_limit_exceeded",
        }

    plain_messages = _to_plain_messages(messages)

    # If this is the first executor turn after decomposition, the message list
    # only has the system prompt. Build the full user message now using the
    # decomposition data from state.
    user_msgs = [m for m in plain_messages if m["role"] == "user"]
    if not user_msgs:
        decomposition = state.get("decomposition")
        first_user = _build_initial_messages(state["statement"], decomposition)
        plain_messages = first_user

    response = call_llm(messages=plain_messages, temperature=0.0)
    content = response["content"] or ""

    # --- Tool call ---
    if response["tool_call"]:
        tc = response["tool_call"]
        # Validate tool name before accepting
        from src.tools import TOOLS
        if tc.get("name") not in TOOLS:
            # Unknown tool — treat as malformed and repair
            content = f'{{"tool": "{tc.get("name", "")}", "args": {{}}}}'
        else:
            return {
                "messages": [
                    {"role": "assistant", "content": content},
                    {"role": "system", "content": f"__TOOL_CALL__:{json.dumps(tc)}"},
                ],
                "steps": steps,
                "verdict_data": None,
                "finish_reason": FinishReason.CONFIDENT,
                "error": None,
            }

    # --- Try to parse a verdict from the text ---
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
        "Respond with ONLY valid JSON — either a tool call or a verdict.\n\n"
        "Tool call format:\n"
        '{"tool": "search_filings", "args": {"query": "..."}}\n\n'
        "Verdict format:\n"
        '{"verdict": "SUPPORTED", "explanation": "...", "evidence": [{"filing_id": "10k_fy2024", "page_number": 68, "passage": "exact quote", "section": null}]}\n\n'
        "verdict must be one of: SUPPORTED, CONTRADICTED, OUTDATED, NOT_ENOUGH_INFO"
    )
    repair_messages = plain_messages + [
        {"role": "assistant", "content": content},
        {"role": "user", "content": repair_prompt},
    ]
    repair_response = call_llm(messages=repair_messages, temperature=0.0)
    repair_content = repair_response["content"] or ""

    if repair_response["tool_call"]:
        tc = repair_response["tool_call"]
        from src.tools import TOOLS
        if tc.get("name") in TOOLS:
            return {
                "messages": [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": repair_prompt},
                    {"role": "assistant", "content": repair_content},
                    {"role": "system", "content": f"__TOOL_CALL__:{json.dumps(tc)}"},
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

    # Both attempts failed — give up gracefully
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


# ---------------------------------------------------------------------------
# Tool Node
# ---------------------------------------------------------------------------
def tool_node(state: AgentState) -> dict:
    """
    Execute the tool encoded in the last __TOOL_CALL__ sentinel message.
    Returns tool result as a user message. Handles tool errors gracefully.
    """
    messages = state["messages"]

    # Find the last sentinel
    tc = None
    for msg in reversed(messages):
        content = (
            msg.get("content", "") if isinstance(msg, dict)
            else getattr(msg, "content", "")
        )
        if content and isinstance(content, str) and content.startswith("__TOOL_CALL__:"):
            try:
                tc = json.loads(content[len("__TOOL_CALL__:"):])
            except json.JSONDecodeError:
                pass
            break

    if not tc:
        return {
            "messages": [{
                "role": "user",
                "content": "Tool error: no pending tool call found. Please call search_filings to continue.",
            }],
        }

    tool_name = tc.get("name", "")
    tool_args = tc.get("arguments", {})

    # Run tool — run_tool already handles unknown tool names
    result = run_tool(tool_name, tool_args)

    # Surface tool errors clearly so the executor can respond to them
    if isinstance(result, dict) and "error" in result:
        result_str = json.dumps({"error": result["error"], "tool": tool_name})
    else:
        result_str = json.dumps(result, ensure_ascii=False)
        # Truncate very long results
        if len(result_str) > 6000:
            result_str = result_str[:6000] + "\n... [truncated]"

    return {
        "messages": [{
            "role": "user",
            "content": f"Tool result for {tool_name}:\n{result_str}",
        }],
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def route_after_executor(state: AgentState) -> str:
    if state.get("verdict_data"):
        return END
    if state["steps"] >= MAX_STEPS:
        return END
    # Check for pending tool call sentinel
    for msg in reversed(state["messages"]):
        content = (
            msg.get("content", "") if isinstance(msg, dict)
            else getattr(msg, "content", "")
        )
        if content and isinstance(content, str) and content.startswith("__TOOL_CALL__:"):
            return "tool_node"
    return END


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_plain_messages(messages: list) -> list[dict]:
    """Convert LangGraph message objects to plain dicts. Strip sentinels."""
    type_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "user"}
    plain = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "") or ""
        else:
            role = type_map.get(getattr(m, "type", "human"), "user")
            content = getattr(m, "content", "") or ""
        if isinstance(content, str) and content.startswith("__TOOL_CALL__:"):
            continue
        plain.append({"role": role, "content": content})
    return plain


def _is_valid_verdict(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    valid = {"SUPPORTED", "CONTRADICTED", "OUTDATED", "NOT_ENOUGH_INFO"}
    return (
        data.get("verdict") in valid
        and "explanation" in data
        and "evidence" in data
    )


def _build_initial_messages(
    statement: str,
    decomposition: dict | None,
    recalled: list[dict] | None = None,
) -> list[dict]:
    """Build the executor's initial message list, optionally enriched by decomposition and memory."""
    context = ""

    # --- Memory recall context ---
    if recalled:
        context += "\n\nPreviously found relevant evidence (from memory — verify before using):\n"
        for i, mem in enumerate(recalled, 1):
            context += (
                f"{i}. [{mem['filing_id']} p{mem['page_number']}] "
                f"Summary: {mem['summary']}\n"
                f"   Passage: {mem['passage'][:200]}\n"
            )
        context += (
            "Use these passages if they directly support your verdict, "
            "but still search for additional evidence to confirm.\n"
        )

    if decomposition and not decomposition.get("_fallback"):
        filings = decomposition.get("relevant_filings", [])
        context += (
            f"\n\nClaim analysis:\n"
            f"- Topic: {decomposition.get('topic', '')}\n"
            f"- Time scope: {decomposition.get('time_scope', 'not specified')}\n"
            f"- Key metric/entity: {decomposition.get('metric_or_entity', '')}\n"
            f"- Suggested search terms: {', '.join(decomposition.get('search_terms', []))}\n"
            f"- Filings to check: {', '.join(filings)}\n"
        )
        if len(filings) > 1:
            context += (
                f"\nIMPORTANT: Search ALL of these filings: {', '.join(filings)}. "
                f"Compare the value across filings. If an older filing matches the "
                f"statement but a newer filing shows a different value, the verdict is OUTDATED.\n"
            )

    # Add hint for "latest" statements — but NOT for specific dated claims
    stmt_lower = statement.lower()
    latest_keywords = ["latest", "current", "most recent", "stands at", "is now"]
    has_specific_date = bool(__import__("re").search(r"as of\s+\w+ \d{1,2},?\s+\d{4}", stmt_lower))
    if any(w in stmt_lower for w in latest_keywords) and not has_specific_date:
        context += (
            "\nNOTE: This statement uses 'latest' or 'current'. "
            "You MUST check the newest filing (10q_oct2024) to see if the value changed. "
            "If an older filing matches but 10q_oct2024 shows a different value, use OUTDATED.\n"
        )

    return [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Verify this statement against the CrowdStrike SEC filings:\n\n"
                f"{statement}{context}\n\n"
                f"Start by calling search_filings."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Build two graphs: with and without decomposer
# ---------------------------------------------------------------------------
def _build_graph(with_decomposer: bool) -> Any:
    builder = StateGraph(AgentState)
    builder.add_node("executor_node", executor_node)
    builder.add_node("tool_node", tool_node)

    if with_decomposer:
        builder.add_node("decomposer_node", decomposer_node)
        builder.set_entry_point("decomposer_node")
        builder.add_edge("decomposer_node", "executor_node")
    else:
        builder.set_entry_point("executor_node")

    builder.add_conditional_edges(
        "executor_node",
        route_after_executor,
        {"tool_node": "tool_node", END: END},
    )
    builder.add_edge("tool_node", "executor_node")
    return builder.compile()


_GRAPH_NO_DECOMP  = _build_graph(with_decomposer=False)
_GRAPH_WITH_DECOMP = _build_graph(with_decomposer=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_check(
    statement: str,
    use_decomposer: bool = True,
    use_memory: bool = False,
) -> VerificationResult:
    """
    Check a statement against the filings.

    Args:
        statement:      The claim to verify
        use_decomposer: Run Decomposer before Executor (default True)
        use_memory:     Use evidence memory — recall before, store after (default False)

    Returns:
        VerificationResult
    """
    reset_usage()
    reset_counters()

    if use_memory:
        from src.memory import recall, store, reset_session_stats
        reset_session_stats()
        recalled = recall(statement, top_k=3)
    else:
        recalled = []

    graph = _GRAPH_WITH_DECOMP if use_decomposer else _GRAPH_NO_DECOMP

    initial_messages = _build_initial_messages(statement, decomposition=None, recalled=recalled)

    initial_state: AgentState = {
        "statement": statement,
        "decomposition": None,
        "messages": initial_messages,
        "steps": 0,
        "verdict_data": None,
        "finish_reason": FinishReason.CONFIDENT,
        "error": None,
    }

    if use_decomposer:
        initial_state["messages"] = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        ]

    final_state = graph.invoke(initial_state)
    result = _build_result(statement, final_state)

    # Store evidence after a confident verdict
    if use_memory and result.finish_reason == FinishReason.CONFIDENT and result.evidence:
        from src.memory import store, get_stats
        stored = store(result.evidence, statement)
        # Update memory_hits in usage
        from src.memory import get_stats
        stats = get_stats()
        result.usage.memory_hits = stats["session_hits"]

    return result


FILING_ORDER = ["10k_fy2024", "10q_jul2024", "10q_oct2024"]


def _check_outdated(
    statement: str,
    evidence_list: list[Evidence],
    current_explanation: str,
    current_verdict: Verdict = Verdict.SUPPORTED,
) -> tuple[Verdict, str | None]:
    """
    Post-process to catch OUTDATED cases where the LLM missed a temporal change.

    Cases handled:
    1. Statement says "latest X is Y", evidence from newest filing shows Z ≠ Y → OUTDATED
    2. Statement says "latest X is Y", evidence from older filing shows Y → OUTDATED

    Returns (current_verdict, None) if no OUTDATED detected — does NOT change the verdict.
    """
    stmt_lower = statement.lower()
    latest_keywords = ["latest", "current", "most recent", "stands at", "is now"]
    has_latest = any(kw in stmt_lower for kw in latest_keywords)

    # Fix 2: If the statement has a specific date (e.g. "as of January 31, 2024"),
    # do NOT apply OUTDATED logic — it's a historical claim, not a "latest" claim
    import re as _re
    has_specific_date = bool(_re.search(r"as of\s+\w+ \d{1,2},?\s+\d{4}", stmt_lower))
    if has_specific_date:
        has_latest = False

    if not evidence_list or not has_latest:
        return current_verdict, None

    # Extract numbers from the statement
    stmt_figures = extract_figures_from_text(statement)
    if not stmt_figures:
        return current_verdict, None

    # Find the newest filing cited in evidence
    cited_filings = [ev.filing_id for ev in evidence_list if ev.filing_id in FILING_ORDER]
    if not cited_filings:
        return current_verdict, None

    newest_cited = max(cited_filings, key=lambda f: FILING_ORDER.index(f))

    # Case 1: evidence from newest filing but numbers don't match statement
    if newest_cited == FILING_ORDER[-1]:
        for ev in evidence_list:
            ev_figures = extract_figures_from_text(ev.passage)
            # Also try normalizing passage figures as "in thousands" (common in filings)
            ev_figures_thousands = []
            for ef in ev_figures:
                from src.numeric import normalize_to_dollars
                ev_figures_thousands.append({
                    **ef,
                    "value": normalize_to_dollars(ef["value"], "thousands")
                })
            all_ev_figures = ev_figures + ev_figures_thousands

            for sf in stmt_figures:
                for ef in all_ev_figures:
                    if ef["value"] == 0:
                        continue
                    if not figures_agree(str(sf["value"]), str(ef["value"]), tolerance=0.02):
                        # The latest filing has a different value.
                        # This means the statement's figure was likely true in an older filing
                        # but the newest filing has since changed it → OUTDATED.
                        return (
                            Verdict.OUTDATED,
                            f"The latest filing (10q_oct2024) reports a different value than "
                            f"the statement claims. The statement's figure likely matched an "
                            f"older filing but has since been updated. {current_explanation}",
                        )
        return current_verdict, None

    # Case 2: evidence from older filing — flag as OUTDATED since there's a newer filing
    return (
        Verdict.OUTDATED,
        f"The statement's figure matches {newest_cited} (an older filing), "
        f"but the newest filing (10q_oct2024) may report a different value. "
        f"Original explanation: {current_explanation}",
    )


def _build_result(statement: str, final_state: dict) -> VerificationResult:
    """Convert final graph state to a VerificationResult."""
    verdict_data = final_state.get("verdict_data") or {
        "verdict": Verdict.NOT_ENOUGH_INFO,
        "explanation": "No verdict was reached.",
        "evidence": [],
    }

    try:
        verdict = Verdict(verdict_data.get("verdict", "NOT_ENOUGH_INFO"))
    except ValueError:
        verdict = Verdict.NOT_ENOUGH_INFO

    evidence_list = []
    for ev in verdict_data.get("evidence", []):
        if not isinstance(ev, dict):
            continue
        # Validate filing_id
        filing_id = ev.get("filing_id", "unknown")
        if filing_id not in {"10k_fy2024", "10q_jul2024", "10q_oct2024"}:
            filing_id = "unknown"
        try:
            page_number = int(ev.get("page_number", 0))
        except (ValueError, TypeError):
            page_number = 0
        passage = ev.get("passage", "")
        # Reject empty, very short, or clearly hallucinated passages
        # Real filing quotes don't start with "The filing does not" or "No information"
        hallucination_markers = [
            "the filing does not",
            "no information",
            "not mentioned",
            "not disclosed",
            "cannot be found",
            "there is no",
            "this information is not",
        ]
        passage_lower = passage.lower().strip()
        if not passage or len(passage.strip()) < 15:
            continue
        if any(passage_lower.startswith(m) for m in hallucination_markers):
            continue
        evidence_list.append(Evidence(
            filing_id=filing_id,
            page_number=page_number,
            passage=passage,
            section=ev.get("section"),
        ))

    # Post-process: catch cases where the LLM missed OUTDATED.
    if verdict == Verdict.SUPPORTED and evidence_list:
        verdict, explanation_override = _check_outdated(
            statement, evidence_list, verdict_data.get("explanation", ""), verdict
        )
        if explanation_override:
            verdict_data["explanation"] = explanation_override

    # Also check CONTRADICTED → OUTDATED (if statement says "latest" but
    # newest filing shows a different value, it's OUTDATED not CONTRADICTED)
    if verdict == Verdict.CONTRADICTED and evidence_list:
        verdict, explanation_override = _check_outdated(
            statement, evidence_list, verdict_data.get("explanation", ""), verdict
        )
        if explanation_override:
            verdict_data["explanation"] = explanation_override

    llm_usage = get_usage()
    tool_usage = get_counters()
    usage = Usage(
        llm_calls=llm_usage["llm_calls"],
        input_tokens=llm_usage["input_tokens"],
        output_tokens=llm_usage["output_tokens"],
        tool_calls=tool_usage["total"],
        memory_hits=0,
    )

    try:
        finish_reason = FinishReason(final_state.get("finish_reason", FinishReason.CONFIDENT))
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


def run_check_stream(statement: str, use_decomposer: bool = True, use_memory: bool = False):
    """
    Generator that yields step events as the agent works, then the final result.
    """
    reset_usage()
    reset_counters()

    if use_memory:
        from src.memory import recall, reset_session_stats
        reset_session_stats()
        recalled = recall(statement, top_k=3)
    else:
        recalled = []

    graph = _GRAPH_WITH_DECOMP if use_decomposer else _GRAPH_NO_DECOMP

    if use_decomposer:
        initial_messages = [{"role": "system", "content": EXECUTOR_SYSTEM_PROMPT}]
    else:
        initial_messages = _build_initial_messages(statement, decomposition=None, recalled=recalled)

    initial_state: AgentState = {
        "statement": statement,
        "decomposition": None,
        "messages": initial_messages,
        "steps": 0,
        "verdict_data": None,
        "finish_reason": FinishReason.CONFIDENT,
        "error": None,
    }

    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "decomposer_node":
                decomp = node_output.get("decomposition", {})
                yield {
                    "type": "step",
                    "step": 0,
                    "action": "decomposed",
                    "detail": f"topic={decomp.get('topic', '')} terms={decomp.get('search_terms', [])}",
                }
            elif node_name == "executor_node":
                if node_output.get("verdict_data"):
                    yield {
                        "type": "step",
                        "step": node_output.get("steps", 0),
                        "action": "verdict",
                        "detail": node_output["verdict_data"].get("verdict"),
                    }
                else:
                    msgs = node_output.get("messages", [])
                    tool_name = None
                    for m in msgs:
                        c = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
                        if c and c.startswith("__TOOL_CALL__:"):
                            try:
                                tc = json.loads(c[len("__TOOL_CALL__:"):])
                                tool_name = tc.get("name")
                            except Exception:
                                pass
                    yield {
                        "type": "step",
                        "step": node_output.get("steps", 0),
                        "action": "tool_call",
                        "detail": tool_name or "unknown",
                    }
            elif node_name == "tool_node":
                msgs = node_output.get("messages", [])
                content = ""
                for m in msgs:
                    content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
                yield {
                    "type": "step",
                    "step": 0,
                    "action": "tool_result",
                    "detail": content[:120] if content else "",
                }

    result = run_check(statement, use_decomposer=use_decomposer, use_memory=use_memory)
    yield {"type": "result", "data": result.model_dump()}
