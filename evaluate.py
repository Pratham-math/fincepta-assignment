"""
evaluate.py — Run all 20 test statements and save predictions.json

Configs:
  A: use_decomposer=True  (full system)
  B: use_decomposer=False (single-agent baseline, no decomposer)

Usage:
    python evaluate.py
    python evaluate.py --config A   # run only config A
    python evaluate.py --config B   # run only config B

Output files:
    predictions.json          (Config A — required by assignment)
    predictions_no_decomp.json (Config B — ablation)
    evaluation_summary.json   (aggregate numbers for both configs)
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from src.agent import run_check

DATA_DIR = Path("data")


def run_config(tests: list[dict], use_decomposer: bool, label: str) -> dict:
    """Run all tests under one config. Returns results dict."""
    results = []
    correct = 0
    total_llm = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_tools = 0
    errors = 0
    step_limits = 0

    verdict_counts = {"SUPPORTED": 0, "CONTRADICTED": 0, "OUTDATED": 0, "NOT_ENOUGH_INFO": 0}
    correct_by_verdict = {"SUPPORTED": 0, "CONTRADICTED": 0, "OUTDATED": 0, "NOT_ENOUGH_INFO": 0}

    print(f"\n{'='*70}")
    print(f"CONFIG {label}: decomposer={'ON' if use_decomposer else 'OFF'}")
    print(f"{'='*70}")

    for i, t in enumerate(tests):
        sid       = t["id"]
        statement = t["statement"]
        expected  = t["expected_verdict"]

        print(f"\n[{i+1:02d}/20] {sid} | expected={expected}")
        print(f"  {statement[:85]}")

        start = time.time()
        try:
            result = run_check(statement, use_decomposer=use_decomposer, use_memory=False)
            elapsed = round(time.time() - start, 1)

            actual = result.verdict.value
            ok = actual == expected
            correct += ok
            verdict_counts[actual] = verdict_counts.get(actual, 0) + 1
            if ok:
                correct_by_verdict[expected] = correct_by_verdict.get(expected, 0) + 1

            total_llm   += result.usage.llm_calls
            total_input_tokens  += result.usage.input_tokens
            total_output_tokens += result.usage.output_tokens
            total_tools += result.usage.tool_calls

            if result.finish_reason.value == "step_limit":
                step_limits += 1

            print(f"  → {actual} {'✓' if ok else '✗'}  |  LLM={result.usage.llm_calls}  tools={result.usage.tool_calls}  {elapsed}s")
            if result.evidence:
                ev = result.evidence[0]
                print(f"     [{ev.filing_id} p{ev.page_number}] {ev.passage[:80]}")
            if not ok:
                print(f"     Explanation: {result.explanation[:120]}")

            entry = {
                "id": sid,
                "statement": statement,
                "expected_verdict": expected,
                "system_verdict": actual,
                "correct": ok,
                "explanation": result.explanation,
                "evidence": [
                    {
                        "filing_id": ev.filing_id,
                        "page_number": ev.page_number,
                        "passage": ev.passage,
                        "section": ev.section,
                    }
                    for ev in result.evidence
                ],
                "usage": {
                    "llm_calls": result.usage.llm_calls,
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "tool_calls": result.usage.tool_calls,
                },
                "finish_reason": result.finish_reason.value,
            }

        except Exception as e:
            elapsed = round(time.time() - start, 1)
            errors += 1
            print(f"  ERROR: {e}")
            entry = {
                "id": sid,
                "statement": statement,
                "expected_verdict": expected,
                "system_verdict": "ERROR",
                "correct": False,
                "explanation": str(e),
                "evidence": [],
                "usage": {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "tool_calls": 0},
                "finish_reason": "error",
            }

        results.append(entry)

    # Summary
    accuracy = correct / len(tests) * 100
    print(f"\n{'='*70}")
    print(f"CONFIG {label} SUMMARY")
    print(f"  Accuracy:      {correct}/20 ({accuracy:.0f}%)")
    print(f"  Step limits:   {step_limits}")
    print(f"  Errors:        {errors}")
    print(f"  Total LLM calls:    {total_llm}")
    print(f"  Total input tokens: {total_input_tokens}")
    print(f"  Total output tokens:{total_output_tokens}")
    print(f"  Total tool calls:   {total_tools}")
    print(f"\n  Per-verdict accuracy:")
    for v in ["SUPPORTED", "CONTRADICTED", "OUTDATED", "NOT_ENOUGH_INFO"]:
        total_v = sum(1 for t in tests if t["expected_verdict"] == v)
        corr_v  = correct_by_verdict.get(v, 0)
        print(f"    {v:<20} {corr_v}/{total_v}")

    return {
        "config": label,
        "use_decomposer": use_decomposer,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(tests),
        "step_limits": step_limits,
        "errors": errors,
        "llm_calls": total_llm,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "tool_calls": total_tools,
        "per_verdict": {
            v: {
                "correct": correct_by_verdict.get(v, 0),
                "total": sum(1 for t in tests if t["expected_verdict"] == v),
            }
            for v in ["SUPPORTED", "CONTRADICTED", "OUTDATED", "NOT_ENOUGH_INFO"]
        },
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["A", "B", "all"], default="all")
    args = parser.parse_args()

    with open(DATA_DIR / "test_statements.json", encoding="utf-8") as f:
        tests = json.load(f)

    summaries = []

    if args.config in ("A", "all"):
        data_a = run_config(tests, use_decomposer=True, label="A")
        summaries.append({k: v for k, v in data_a.items() if k != "results"})
        # Save predictions.json (assignment requirement)
        with open(DATA_DIR / "predictions.json", "w", encoding="utf-8") as f:
            json.dump(data_a["results"], f, indent=2, ensure_ascii=False)
        print(f"\nSaved predictions.json ({len(data_a['results'])} entries)")

    if args.config in ("B", "all"):
        data_b = run_config(tests, use_decomposer=False, label="B")
        summaries.append({k: v for k, v in data_b.items() if k != "results"})
        with open(DATA_DIR / "predictions_no_decomp.json", "w", encoding="utf-8") as f:
            json.dump(data_b["results"], f, indent=2, ensure_ascii=False)
        print(f"\nSaved predictions_no_decomp.json ({len(data_b['results'])} entries)")

    if args.config == "all" and len(summaries) == 2:
        print(f"\n{'='*70}")
        print("COMPARISON TABLE")
        print(f"{'='*70}")
        print(f"{'Metric':<28} {'Config A (decomp)':<22} {'Config B (no decomp)'}")
        print("-" * 70)
        metrics = [
            ("Accuracy", "accuracy", "{:.0f}%"),
            ("LLM calls", "llm_calls", "{}"),
            ("Input tokens", "input_tokens", "{}"),
            ("Output tokens", "output_tokens", "{}"),
            ("Tool calls", "tool_calls", "{}"),
            ("Step limits", "step_limits", "{}"),
        ]
        for label, key, fmt in metrics:
            va = fmt.format(summaries[0][key])
            vb = fmt.format(summaries[1][key])
            print(f"  {label:<26} {va:<22} {vb}")

    # Save combined summary
    with open(DATA_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSaved evaluation_summary.json")


if __name__ == "__main__":
    main()
