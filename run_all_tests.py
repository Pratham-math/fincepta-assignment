"""
Run all 20 test statements and print a results table.
Usage: python run_all_tests.py
"""
import json
import sys
sys.path.insert(0, ".")

from src.agent import run_check

with open("data/test_statements.json", encoding="utf-8") as f:
    tests = json.load(f)

results = []
passed = 0

for i, t in enumerate(tests):
    sid        = t["id"]
    statement  = t["statement"]
    expected   = t["expected_verdict"]

    print(f"\n[{i+1}/20] {sid} | expected={expected}")
    print(f"  {statement[:90]}")

    try:
        result = run_check(statement, use_decomposer=True)
        actual  = result.verdict.value
        ok      = actual == expected
        passed += ok

        ev_summary = ""
        if result.evidence:
            ev = result.evidence[0]
            ev_summary = f"{ev.filing_id} p{ev.page_number}: {ev.passage[:80]}"

        print(f"  → {actual} {'✓' if ok else '✗'}  |  LLM={result.usage.llm_calls} tools={result.usage.tool_calls}  |  {result.finish_reason.value}")
        if ev_summary:
            print(f"  Evidence: {ev_summary}")
        if not ok:
            print(f"  Explanation: {result.explanation[:150]}")

        results.append({
            "id": sid,
            "statement": statement,
            "expected": expected,
            "got": actual,
            "correct": ok,
            "llm_calls": result.usage.llm_calls,
            "tool_calls": result.usage.tool_calls,
            "finish_reason": result.finish_reason.value,
            "explanation": result.explanation,
            "evidence": [
                {"filing_id": ev.filing_id, "page_number": ev.page_number, "passage": ev.passage}
                for ev in result.evidence
            ],
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            "id": sid, "statement": statement, "expected": expected,
            "got": "ERROR", "correct": False, "error": str(e),
            "llm_calls": 0, "tool_calls": 0, "finish_reason": "error",
            "explanation": "", "evidence": [],
        })

# Summary table
print(f"\n{'='*70}")
print(f"RESULTS: {passed}/20 correct")
print(f"{'='*70}")
print(f"{'ID':<6} {'Expected':<18} {'Got':<18} {'OK'}")
print("-" * 60)
for r in results:
    ok_mark = "✓" if r["correct"] else "✗"
    print(f"{r['id']:<6} {r['expected']:<18} {r['got']:<18} {ok_mark}")

total_llm   = sum(r.get("llm_calls", 0) for r in results)
total_tools = sum(r.get("tool_calls", 0) for r in results)
print(f"\nTotal LLM calls:  {total_llm}")
print(f"Total tool calls: {total_tools}")

# Save results
with open("data/run_all_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nSaved to data/run_all_results.json")
