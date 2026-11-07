"""
evaluate_memory.py — Evaluate the knowledge graph memory.

Runs memory_test_statements.json in order within each group.
For each group, compares:
  - Tool calls WITH memory vs WITHOUT memory
  - LLM calls WITH memory vs WITHOUT memory
  - Accuracy (verdict correctness)

Usage:
    python evaluate_memory.py
"""

import json
from pathlib import Path
from src.agent import run_check
from src.memory import clear_memory, get_stats, reset_session_stats

DATA_DIR = Path("data")

with open(DATA_DIR / "memory_test_statements.json", encoding="utf-8") as f:
    groups = json.load(f)

print(f"Loaded {len(groups)} groups, {sum(len(g['statements']) for g in groups)} statements\n")

# ── Run WITH memory ──────────────────────────────────────────────
print("=" * 70)
print("RUN A: WITH MEMORY (use_memory=True)")
print("=" * 70)

clear_memory()
results_with = {}

for group in groups:
    gid = group["group"]
    print(f"\nGroup: {gid} — {group['description']}")
    group_results = []

    for stmt in group["statements"]:
        sid = stmt["id"]
        statement = stmt["statement"]
        expected = stmt["expected_verdict"]
        role = stmt["memory_role"]

        reset_session_stats()
        result = run_check(statement, use_decomposer=False, use_memory=True)
        stats = get_stats()

        ok = result.verdict.value == expected
        group_results.append({
            "id": sid,
            "role": role,
            "expected": expected,
            "got": result.verdict.value,
            "correct": ok,
            "llm_calls": result.usage.llm_calls,
            "tool_calls": result.usage.tool_calls,
            "memory_hits": stats["session_hits"],
        })

        icon = "✓" if ok else "✗"
        print(f"  {icon} [{sid}] {role:<25} expected={expected:<18} got={result.verdict.value:<18} "
              f"llm={result.usage.llm_calls} tools={result.usage.tool_calls} mem_hits={stats['session_hits']}")

    results_with[gid] = group_results

# ── Run WITHOUT memory ───────────────────────────────────────────
print("\n\n" + "=" * 70)
print("RUN B: WITHOUT MEMORY (use_memory=False)")
print("=" * 70)

clear_memory()
results_without = {}

for group in groups:
    gid = group["group"]
    print(f"\nGroup: {gid} — {group['description']}")
    group_results = []

    for stmt in group["statements"]:
        sid = stmt["id"]
        statement = stmt["statement"]
        expected = stmt["expected_verdict"]
        role = stmt["memory_role"]

        result = run_check(statement, use_decomposer=False, use_memory=False)

        ok = result.verdict.value == expected
        group_results.append({
            "id": sid,
            "role": role,
            "expected": expected,
            "got": result.verdict.value,
            "correct": ok,
            "llm_calls": result.usage.llm_calls,
            "tool_calls": result.usage.tool_calls,
            "memory_hits": 0,
        })

        icon = "✓" if ok else "✗"
        print(f"  {icon} [{sid}] {role:<25} expected={expected:<18} got={result.verdict.value:<18} "
              f"llm={result.usage.llm_calls} tools={result.usage.tool_calls}")

    results_without[gid] = group_results

# ── Comparison table ─────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("COMPARISON: WITH vs WITHOUT MEMORY")
print("=" * 70)

total_llm_with = total_llm_without = 0
total_tools_with = total_tools_without = 0
correct_with = correct_without = 0
total = 0

print(f"\n{'ID':<6} {'Role':<25} {'Exp':<18} {'W_mem':<12} {'No_mem':<12} {'LLM↓':<8} {'Tools↓'}")
print("-" * 85)

for group in groups:
    gid = group["group"]
    for i, stmt in enumerate(group["statements"]):
        sid = stmt["id"]
        rw = results_with[gid][i]
        rn = results_without[gid][i]

        total += 1
        correct_with    += rw["correct"]
        correct_without += rn["correct"]
        total_llm_with    += rw["llm_calls"]
        total_llm_without += rn["llm_calls"]
        total_tools_with    += rw["tool_calls"]
        total_tools_without += rn["tool_calls"]

        llm_diff   = rn["llm_calls"]   - rw["llm_calls"]
        tools_diff = rn["tool_calls"]  - rw["tool_calls"]
        w_verdict  = "✓" if rw["correct"] else "✗"
        n_verdict  = "✓" if rn["correct"] else "✗"

        w_str = f"{w_verdict}{rw['got'][:10]}"
        n_str = f"{n_verdict}{rn['got'][:10]}"
        llm_str   = f"+{llm_diff}"   if llm_diff > 0   else str(llm_diff)
        tools_str = f"+{tools_diff}" if tools_diff > 0 else str(tools_diff)

        print(f"{sid:<6} {rw['role']:<25} {rw['expected']:<18} {w_str:<12} {n_str:<12} {llm_str:<8} {tools_str}")

print("-" * 85)
print(f"\n{'TOTALS':}")
print(f"  Accuracy     WITH memory: {correct_with}/{total} ({100*correct_with//total}%)")
print(f"  Accuracy  WITHOUT memory: {correct_without}/{total} ({100*correct_without//total}%)")
print(f"  LLM calls    WITH memory: {total_llm_with}")
print(f"  LLM calls WITHOUT memory: {total_llm_without}  (saved: {total_llm_without - total_llm_with})")
print(f"  Tool calls   WITH memory: {total_tools_with}")
print(f"  Tool calls WITHOUT memory: {total_tools_without}  (saved: {total_tools_without - total_tools_with})")

# ── Save results ─────────────────────────────────────────────────
output = {
    "with_memory":    {gid: results_with[gid]    for gid in results_with},
    "without_memory": {gid: results_without[gid] for gid in results_without},
    "summary": {
        "accuracy_with":       f"{correct_with}/{total}",
        "accuracy_without":    f"{correct_without}/{total}",
        "llm_calls_with":      total_llm_with,
        "llm_calls_without":   total_llm_without,
        "llm_calls_saved":     total_llm_without - total_llm_with,
        "tool_calls_with":     total_tools_with,
        "tool_calls_without":  total_tools_without,
        "tool_calls_saved":    total_tools_without - total_tools_with,
    }
}

with open(DATA_DIR / "memory_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved to data/memory_evaluation.json")
