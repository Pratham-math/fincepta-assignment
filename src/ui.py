"""
ui.py — Streamlit reviewer interface for the Filing Fact-Checker.

Run with:
  streamlit run src/ui.py

Requires the FastAPI server to be running on localhost:8000:
  uvicorn src.api:app --reload --port 8000
"""

import json
import time
from pathlib import Path

import httpx
import streamlit as st

API_URL = "http://localhost:8000"
TEST_FILE = Path("data/test_statements.json")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Filing Fact-Checker",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Filing Fact-Checker")
st.caption("Verifies statements about CrowdStrike Holdings against its official SEC filings.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    use_decomposer = st.toggle("Use Decomposer", value=True,
                                help="Run claim decomposition before the executor.")
    st.divider()
    st.markdown("**Filings indexed:**")
    st.markdown("- 10-K FY2024 (Jan 31 2024)")
    st.markdown("- 10-Q Q2FY2025 (Jul 31 2024)")
    st.markdown("- 10-Q Q3FY2025 (Oct 31 2024)")
    st.divider()
    st.markdown("**API:** `localhost:8000`")

# ---------------------------------------------------------------------------
# Verdict styling
# ---------------------------------------------------------------------------
VERDICT_COLORS = {
    "SUPPORTED":       "✅ SUPPORTED",
    "CONTRADICTED":    "❌ CONTRADICTED",
    "OUTDATED":        "⚠️ OUTDATED",
    "NOT_ENOUGH_INFO": "❓ NOT ENOUGH INFO",
}

VERDICT_BG = {
    "SUPPORTED":       "#d4edda",
    "CONTRADICTED":    "#f8d7da",
    "OUTDATED":        "#fff3cd",
    "NOT_ENOUGH_INFO": "#e2e3e5",
}


def render_verdict(verdict: str, explanation: str, evidence: list, usage: dict, finish_reason: str):
    color = VERDICT_BG.get(verdict, "#ffffff")
    label = VERDICT_COLORS.get(verdict, verdict)
    st.markdown(
        f'<div style="background:{color};padding:12px;border-radius:8px;margin-bottom:8px">'
        f'<b style="font-size:1.1em">{label}</b><br>{explanation}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if evidence:
        st.markdown("**Evidence:**")
        for i, ev in enumerate(evidence, 1):
            with st.expander(f"[{i}] {ev.get('filing_id','?')} — page {ev.get('page_number','?')}  |  {ev.get('section') or ''}"):
                st.markdown(f"> {ev.get('passage','')}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LLM calls", usage.get("llm_calls", 0))
    col2.metric("Tool calls", usage.get("tool_calls", 0))
    col3.metric("Tokens in", usage.get("input_tokens", 0))
    col4.metric("Tokens out", usage.get("output_tokens", 0))

    if finish_reason == "step_limit":
        st.warning("⚠️ Agent reached step limit — verdict may be incomplete.")


def stream_and_render(statement: str, log_container, result_container):
    """Call the API, stream steps into log_container, render result in result_container."""
    steps_log = []
    result_data = None

    payload = {"statement": statement, "use_decomposer": use_decomposer}

    try:
        with httpx.stream("POST", f"{API_URL}/check", json=payload, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "step":
                    action = event.get("action", "")
                    detail = event.get("detail", "")
                    step_n = event.get("step", 0)
                    if action == "decomposed":
                        msg = f"🔍 Decomposed: {detail}"
                    elif action == "tool_call":
                        msg = f"🔧 Step {step_n}: calling `{detail}`"
                    elif action == "tool_result":
                        msg = f"📥 Tool result: {detail[:80]}"
                    elif action == "verdict":
                        msg = f"⚖️ Verdict ready: {detail}"
                    else:
                        msg = f"→ {action}: {detail}"
                    steps_log.append(msg)
                    with log_container:
                        for s in steps_log:
                            st.text(s)

                elif event.get("type") == "result":
                    result_data = event.get("data", {})

                elif event.get("type") == "error":
                    with result_container:
                        st.error(f"Agent error: {event.get('detail')}")
                    return

    except httpx.ConnectError:
        with result_container:
            st.error("Cannot connect to API. Make sure the server is running:\n\n`uvicorn src.api:app --reload --port 8000`")
        return
    except Exception as e:
        with result_container:
            st.error(f"Request failed: {e}")
        return

    if result_data:
        with result_container:
            render_verdict(
                verdict=result_data.get("verdict", "NOT_ENOUGH_INFO"),
                explanation=result_data.get("explanation", ""),
                evidence=result_data.get("evidence", []),
                usage=result_data.get("usage", {}),
                finish_reason=result_data.get("finish_reason", "confident"),
            )


# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["Single Check", "Run Test Set"])

# ── Single Check ────────────────────────────────────────────────────────────
with tab_single:
    statement_input = st.text_area(
        "Statement to verify",
        height=80,
        placeholder="e.g. CrowdStrike generated approximately $3.06 billion in total revenue in fiscal 2024.",
    )

    if st.button("Check", type="primary", key="single_check"):
        if not statement_input.strip():
            st.warning("Please enter a statement.")
        else:
            st.divider()
            st.markdown("**Agent steps:**")
            log_box = st.container()
            st.divider()
            st.markdown("**Result:**")
            result_box = st.container()
            with st.spinner("Verifying..."):
                stream_and_render(statement_input.strip(), log_box, result_box)

# ── Batch Test Set ───────────────────────────────────────────────────────────
with tab_batch:
    if not TEST_FILE.exists():
        st.warning("No test statements found. Save your statements to `data/test_statements.json`.")
    else:
        with open(TEST_FILE, encoding="utf-8") as f:
            test_statements = json.load(f)

        st.markdown(f"**{len(test_statements)} test statements loaded.**")

        # Preview table
        with st.expander("Preview statements"):
            for t in test_statements:
                st.markdown(f"**{t['id']}** `{t['expected_verdict']}` — {t['statement'][:90]}")

        if st.button("Run All", type="primary", key="batch_run"):
            st.divider()
            progress = st.progress(0)
            status_text = st.empty()

            results = []
            correct = 0

            for i, t in enumerate(test_statements):
                status_text.text(f"[{i+1}/{len(test_statements)}] {t['id']} — {t['statement'][:60]}...")

                st.markdown(f"---\n**[{i+1}] {t['id']}** — Expected: `{t['expected_verdict']}`")
                st.caption(t["statement"])
                log_box = st.container()
                result_box = st.container()

                stream_and_render(t["statement"], log_box, result_box)
                progress.progress((i + 1) / len(test_statements))

            status_text.text("Done!")

        # Show existing predictions.json if available
        pred_file = Path("data/predictions.json")
        if pred_file.exists():
            st.divider()
            st.markdown("**Saved predictions (from last evaluate.py run):**")
            with open(pred_file, encoding="utf-8") as f:
                preds = json.load(f)

            correct = sum(1 for p in preds if p.get("correct"))
            st.metric("Accuracy", f"{correct}/{len(preds)} ({correct/len(preds)*100:.0f}%)")

            rows = []
            for p in preds:
                ok = "✓" if p.get("correct") else "✗"
                rows.append({
                    "ID": p["id"],
                    "Expected": p["expected_verdict"],
                    "Got": p["system_verdict"],
                    "OK": ok,
                    "LLM": p.get("usage", {}).get("llm_calls", 0),
                    "Tools": p.get("usage", {}).get("tool_calls", 0),
                })
            st.dataframe(rows, use_container_width=True)
