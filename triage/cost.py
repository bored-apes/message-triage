"""Cost per 1,000 messages.

    python -m triage.cost                       # estimate from stated assumptions
    python -m triage.cost out/results.jsonl     # measured, from a real run

The estimate is token arithmetic, so the assumptions are spelled out here
rather than buried in a spreadsheet. Replace them with measured numbers once a
real run exists (the second form does that automatically).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import PRICES

# ---- Assumptions (per message) ---------------------------------------------
# Measured on the 25-message live run (claude-opus-5-5, effort=low): cached prefix 3,907
# tokens/call, uncached input 156, output 490 (incl. thinking). Budgets below add headroom.
SYSTEM_TOKENS = 3900        # system prompt + structured-output schema overhead (measured)
USER_TOKENS = 250           # metadata envelope (~80) + message body. This file averages ~35 tokens;
                            # real email bodies run longer, so 170 is budgeted for the body.
OUTPUT_TOKENS = 600         # measured 490 avg (JSON + low-effort thinking); +20% for longer real messages
OUTPUT_TOKENS_HAIKU = 500   # no thinking on Haiku
RETRY_RATE = 0.02           # invalid/truncated output re-asked once
MESSAGES_PER_DAY = 10_000
CACHE_TTL_MIN = 5
# Min cacheable prefix: 512 tokens on Opus 5.5 / Sonnet 5.5, 4096 on Haiku 4.5,
# so our ~3.9k-token prefix caches on the 5.x models and NOT on Haiku 4.5.
CACHEABLE = {"claude-opus-5-5": True, "claude-sonnet-5-5": True, "claude-haiku-4-5": False}


def estimate(model: str) -> dict:
    inp, out, cr, cw = (p / 1e6 for p in PRICES[model])
    out_tokens = OUTPUT_TOKENS_HAIKU if model.startswith("claude-haiku") else OUTPUT_TOKENS
    if CACHEABLE[model]:
        per_msg = SYSTEM_TOKENS * cr + USER_TOKENS * inp + out_tokens * out
        # Worst case one cache write per TTL window per day (traffic is ~7 msg/min, so the cache stays warm).
        writes_per_day = 24 * 60 / CACHE_TTL_MIN
        cache_write_per_msg = writes_per_day * SYSTEM_TOKENS * (cw - cr) / MESSAGES_PER_DAY
    else:
        per_msg = (SYSTEM_TOKENS + USER_TOKENS) * inp + out_tokens * out
        cache_write_per_msg = 0.0
    per_msg = per_msg * (1 + RETRY_RATE) + cache_write_per_msg
    return {
        "model": model,
        "usd_per_1000": round(per_msg * 1000, 2),
        "usd_per_day_at_10k": round(per_msg * MESSAGES_PER_DAY, 2),
        "usd_per_month_at_10k": round(per_msg * MESSAGES_PER_DAY * 30, 0),
        "usd_per_1000_batch_api": round(per_msg * 1000 * 0.5, 2),
    }


def measured(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    llm = [r for r in rows if r["meta"]["classifier"] == "llm" or r["meta"]["cost_usd"] > 0]
    if not llm:
        return {"error": "no model-classified rows in this file (offline run?)"}
    n = len(llm)
    cost = sum(r["meta"]["cost_usd"] for r in llm)
    avg = lambda k: round(sum(r["meta"][k] for r in llm) / n, 1)
    return {
        "rows_billed": n,
        "avg_input_tokens": avg("input_tokens"),
        "avg_cache_read_tokens": avg("cache_read_tokens"),
        "avg_cache_write_tokens": avg("cache_write_tokens"),
        "avg_output_tokens": avg("output_tokens"),
        "avg_latency_ms": avg("latency_ms"),
        "usd_per_1000": round(cost / n * 1000, 2),
        "usd_per_day_at_10k": round(cost / n * MESSAGES_PER_DAY, 2),
        "note": "A 25-message run pays the cache write on the first calls; steady state is lower.",
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(measured(Path(sys.argv[1])), indent=2))
    else:
        print(json.dumps([estimate(m) for m in PRICES], indent=2))
