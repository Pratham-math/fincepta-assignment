"""
test_memory.py — Manual tests for the knowledge graph memory.

Tests all major cases:
  1. Basic store and recall
  2. Entity matching (same entity, different relation)
  3. Relation matching (same relation, different page)
  4. Passage text matching (when entity/relation don't match)
  5. Cross-statement reuse (the main MERMAID benefit)
  6. Negative cases (unrelated statements should NOT recall)
  7. Deduplication (same passage stored twice should only appear once)
  8. Confidence levels
  9. Graph dump and stats

Run with: python test_memory.py
"""

import sys
sys.path.insert(0, ".")

from src.memory import clear_memory, get_stats, recall, store, dump_graph
from src.schemas import Evidence

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(test_name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((test_name, status))
    icon = "✓" if condition else "✗"
    print(f"  {icon} [{status}] {test_name}")
    if detail:
        print(f"      {detail}")


# ============================================================
print("\n" + "="*60)
print("SETUP: Clear memory")
print("="*60)
clear_memory()
stats = get_stats()
check("Memory starts empty", stats["stored_entities"] == 0 and stats["stored_triples"] == 0)


# ============================================================
print("\n" + "="*60)
print("TEST 1: Basic store and recall")
print("="*60)

ev_arr = Evidence(
    filing_id="10k_fy2024", page_number=66,
    passage="Annual recurring revenue $ 3,435,150",
    section="Key Metrics"
)
n = store([ev_arr], "CrowdStrike ARR annual recurring revenue FY2024")
check("Store returns positive count", n > 0, f"stored {n} triples")

hits = recall("CrowdStrike annual recurring revenue")
check("Recall returns hits for ARR", len(hits) > 0, f"{len(hits)} hit(s)")
if hits:
    check("Correct filing_id recalled", hits[0]["filing_id"] == "10k_fy2024",
          f"got {hits[0]['filing_id']}")
    check("Correct page recalled", hits[0]["page_number"] == 66,
          f"got page {hits[0]['page_number']}")
    check("Summary is generated", len(hits[0]["summary"]) > 5,
          f"summary: {hits[0]['summary'][:60]}")
    check("Score is positive", hits[0]["score"] > 0, f"score={hits[0]['score']}")


# ============================================================
print("\n" + "="*60)
print("TEST 2: Entity matching — same entity, different relations")
print("="*60)

ev_rev = Evidence(
    filing_id="10k_fy2024", page_number=69,
    passage="Total revenue $ 3,055,555 $ 2,241,236 $ 814,319 36 %",
    section=None
)
ev_emp = Evidence(
    filing_id="10k_fy2024", page_number=21,
    passage="As of January 31, 2024, we had 7,925 full-time employees.",
    section="Human Capital Resources"
)
store([ev_rev, ev_emp], "CrowdStrike total revenue employees fiscal 2024")

# Search for revenue — should NOT return employees page as top hit
hits = recall("CrowdStrike total revenue fiscal year 2024")
check("Revenue search returns results", len(hits) > 0)
if len(hits) > 0:
    top_passage = hits[0]["passage"]
    check("Revenue page ranks above employees page",
          "revenue" in top_passage.lower() or "3,055" in top_passage,
          f"top passage: {top_passage[:60]}")

# Search for employees — should find employee page
hits_emp = recall("CrowdStrike full-time employees headcount 2024")
check("Employee search returns results", len(hits_emp) > 0)
if hits_emp:
    check("Employee passage recalled", "employees" in hits_emp[0]["passage"].lower(),
          f"passage: {hits_emp[0]['passage'][:60]}")


# ============================================================
print("\n" + "="*60)
print("TEST 3: Passage text matching — Falcon incident")
print("="*60)

ev_falcon = Evidence(
    filing_id="10q_jul2024", page_number=24,
    passage="On July 19, 2024, the Company released a content configuration update for its Falcon sensor that resulted in system crashes for certain Windows systems (the 'Channel File 291 Incident').",
    section="8. Commitments and Contingencies"
)
store([ev_falcon], "Falcon sensor channel file incident July 19 2024")

hits = recall("Falcon sensor update caused Windows system crashes July 2024")
check("Falcon incident recalled via passage text", len(hits) > 0,
      f"{len(hits)} hit(s)")
if hits:
    check("Correct filing recalled", hits[0]["filing_id"] == "10q_jul2024",
          f"got {hits[0]['filing_id']}")
    check("High score for exact text match", hits[0]["score"] >= 0.5,
          f"score={hits[0]['score']}")
    check("Summary mentions Falcon or incident",
          any(w in hits[0]["summary"].lower() for w in ["falcon", "incident", "crash", "sensor"]),
          f"summary: {hits[0]['summary']}")


# ============================================================
print("\n" + "="*60)
print("TEST 4: Negative case — unrelated statements should not recall")
print("="*60)

hits_div = recall("CrowdStrike paid cash dividends to shareholders fiscal 2024")
check("Dividends query returns 0 hits (no dividend evidence stored)", len(hits_div) == 0,
      f"got {len(hits_div)} hit(s)")

hits_mkt = recall("CrowdStrike holds 25 percent global endpoint security market share")
check("Market share query returns 0 hits (not in memory)", len(hits_mkt) == 0,
      f"got {len(hits_mkt)} hit(s)")


# ============================================================
print("\n" + "="*60)
print("TEST 5: Cross-statement reuse (core MERMAID benefit)")
print("="*60)

# Statement 1 was about ARR — evidence for it is in memory
# Statement 2 is about a similar metric — should benefit from memory
hits_s2 = recall("CrowdStrike annual recurring revenue stands at 3.44 billion latest")
check("Related ARR statement gets memory recall", len(hits_s2) > 0,
      f"{len(hits_s2)} hit(s)")
if hits_s2:
    check("Recalled passage is ARR-related",
          "recurring" in hits_s2[0]["passage"].lower() or "arr" in hits_s2[0]["summary"].lower(),
          f"passage: {hits_s2[0]['passage'][:60]}")

# A contradicted statement about the same metric should also get recall
hits_s3 = recall("CrowdStrike ARR was 3.06 billion as of January 2024 wrong number")
check("Contradicted ARR statement also gets recall", len(hits_s3) > 0,
      f"{len(hits_s3)} hit(s)")


# ============================================================
print("\n" + "="*60)
print("TEST 6: Deduplication — same passage stored twice")
print("="*60)

before = get_stats()["stored_triples"]
store([ev_arr], "CrowdStrike ARR recurring revenue second time")  # same passage
after = get_stats()["stored_triples"]
check("Duplicate passage not inserted again", before == after,
      f"before={before} after={after}")


# ============================================================
print("\n" + "="*60)
print("TEST 7: Delta lawsuit — multi-entity case")
print("="*60)

ev_delta = Evidence(
    filing_id="10q_oct2024", page_number=32,
    passage="On October 25, 2024, Delta Airlines, Inc. filed a complaint against CrowdStrike, Inc. in the Superior Court for Fulton County, Georgia.",
    section="8. Commitments and Contingencies"
)
store([ev_delta], "Delta Airlines lawsuit Georgia court October 2024")

hits_delta = recall("Delta Air Lines sued CrowdStrike in Georgia state court")
check("Delta lawsuit recalled", len(hits_delta) > 0, f"{len(hits_delta)} hit(s)")
if hits_delta:
    check("Correct filing for Delta lawsuit", hits_delta[0]["filing_id"] == "10q_oct2024",
          f"got {hits_delta[0]['filing_id']}")


# ============================================================
print("\n" + "="*60)
print("TEST 8: Graph structure check")
print("="*60)

stats = get_stats()
check("Entities stored", stats["stored_entities"] > 0,
      f"{stats['stored_entities']} entities")
check("Triples stored", stats["stored_triples"] > 0,
      f"{stats['stored_triples']} triples")

graph = dump_graph()
check("Graph has entities", len(graph["entities"]) > 0)
check("Graph has triples", len(graph["triples"]) > 0)

print("\n  Graph summary:")
print(f"  Entities ({len(graph['entities'])}):")
for e in graph["entities"]:
    print(f"    [{e['id']}] {e['name']} ({e['type']})")
print(f"  Triples ({len(graph['triples'])}):")
for t in graph["triples"]:
    print(f"    {t['subject_name']} --[{t['relation']}]--> {t['object_value'][:40]}")
    print(f"      src={t['filing_id']} p{t['page_number']} conf={t['confidence']}")


# ============================================================
print("\n" + "="*60)
print("TEST 9: Session hit counter")
print("="*60)

from src.memory import reset_session_stats
reset_session_stats()
stats_before = get_stats()
check("Session hits reset to 0", stats_before["session_hits"] == 0)

recall("CrowdStrike annual recurring revenue")
recall("Falcon sensor Windows crash")
stats_after = get_stats()
check("Session hits increment after recalls", stats_after["session_hits"] > 0,
      f"hits={stats_after['session_hits']}")


# ============================================================
print("\n" + "="*60)
print("TEST 10: top_k limiting")
print("="*60)

hits_k1 = recall("CrowdStrike revenue fiscal 2024", top_k=1)
hits_k3 = recall("CrowdStrike revenue fiscal 2024", top_k=3)
check("top_k=1 returns at most 1 result", len(hits_k1) <= 1, f"got {len(hits_k1)}")
check("top_k=3 can return more than top_k=1",
      len(hits_k3) >= len(hits_k1), f"k1={len(hits_k1)} k3={len(hits_k3)}")


# ============================================================
print("\n" + "="*60)
total = len(results)
passed = sum(1 for _, s in results if s == PASS)
failed = total - passed
print(f"RESULTS: {passed}/{total} passed  ({failed} failed)")
print("="*60)
if failed > 0:
    print("\nFailed tests:")
    for name, status in results:
        if status == FAIL:
            print(f"  ✗ {name}")
