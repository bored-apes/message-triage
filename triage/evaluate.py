"""Score a results.jsonl against hand labels.

    python -m triage.evaluate out/results.jsonl [data/expected.json]

Reports per-message pass/fail with the reason, plus the two numbers that
matter most for safety:
  * unsafe_auto: auto_resolve where a human was required (must be 0)
  * missed_urgent: an expected urgent_escalation that was not escalated (must be 0)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ORDER = ["low", "normal", "high", "critical"]


def check(res: dict, exp: dict) -> list:
    fails = []
    got_intents = {i["intent"] for i in res["analysis"]["intents"]}
    for group in exp.get("intents", []):
        if not got_intents & set(group):
            fails.append(f"missing intent one-of {group} (got {sorted(got_intents)})")
    decision = res["routing"]["decision"]
    if decision not in exp["decisions"]:
        fails.append(f"decision {decision} not in {exp['decisions']}")
    if "queue_prefix" in exp and not res["routing"]["primary_queue"].startswith(exp["queue_prefix"]):
        fails.append(f"queue {res['routing']['primary_queue']} != {exp['queue_prefix']}*")
    if "min_priority" in exp and ORDER.index(res["routing"]["priority"]) < ORDER.index(exp["min_priority"]):
        fails.append(f"priority {res['routing']['priority']} < {exp['min_priority']}")
    flags = set(res["analysis"]["risk_flags"])
    for f in exp.get("flags", []):
        if f not in flags:
            fails.append(f"missing flag {f}")
    values = {e["value"] for e in res["analysis"]["entities"]}
    for ent in exp.get("entities", []):
        if ent not in values:
            fails.append(f"missing entity {ent}")
    return fails


def main() -> int:
    results_path = Path(sys.argv[1] if len(sys.argv) > 1 else "out/results.jsonl")
    expected_path = Path(sys.argv[2] if len(sys.argv) > 2 else "data/expected.json")
    expected = {k: v for k, v in json.loads(expected_path.read_text()).items() if not k.startswith("_")}
    results = {}
    for line in results_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            results[r["message_id"]] = r

    passed, unsafe_auto, missed_urgent = 0, 0, 0
    for mid, exp in expected.items():
        res = results.get(mid)
        if res is None:
            print(f"FAIL {mid}: no result")
            continue
        fails = check(res, exp)
        if res["routing"]["decision"] == "auto_resolve" and "auto_resolve" not in exp["decisions"]:
            unsafe_auto += 1
        if "urgent_escalation" in exp["decisions"] and res["routing"]["decision"] != "urgent_escalation":
            missed_urgent += 1
        if fails:
            print(f"FAIL {mid}: " + "; ".join(fails))
        else:
            passed += 1
            print(f"ok   {mid}: {res['routing']['decision']} -> {res['routing']['primary_queue']} ({res['confidence']['score']})")
    print(json.dumps({"passed": passed, "total": len(expected), "unsafe_auto": unsafe_auto, "missed_urgent": missed_urgent}))
    return 0 if unsafe_auto == 0 and missed_urgent == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
