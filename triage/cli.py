"""Command line entry point.

    python -m triage data/messages.json                 # uses Claude if a key is configured
    python -m triage data/messages.json --offline       # deterministic rules only, no API
    python -m triage data/messages.json --out out/      # writes results.jsonl + summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

from . import config
from .ingest import load_messages
from .pipeline import TriagePipeline


def _has_credentials() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def summarize(results, file_issues) -> dict:
    n = len(results)
    cost = sum(r.meta.cost_usd for r in results)
    llm_n = sum(1 for r in results if r.meta.classifier == "llm")
    return {
        "messages": n,
        "file_issues": file_issues,
        "decisions": dict(Counter(r.routing.decision.value for r in results)),
        "priorities": dict(Counter(r.routing.priority.value for r in results)),
        "classifier": dict(Counter(r.meta.classifier for r in results)),
        "confidence_bands": dict(Counter(r.confidence.band for r in results)),
        "queues": dict(Counter(r.routing.primary_queue for r in results)),
        "tokens": {
            "input": sum(r.meta.input_tokens for r in results),
            "output": sum(r.meta.output_tokens for r in results),
            "cache_read": sum(r.meta.cache_read_tokens for r in results),
            "cache_write": sum(r.meta.cache_write_tokens for r in results),
        },
        "cost_usd_total": round(cost, 4),
        "cost_usd_per_1000_messages": round(cost / llm_n * 1000, 2) if llm_n else None,
        "errors": dict(Counter(e.split(":")[0] for r in results for e in r.meta.errors)),
    }


def _print_table(results) -> None:
    print(f"{'id':<10} {'decision':<18} {'pri':<8} {'conf':>5}  {'queue':<34} intents", file=sys.stderr)
    for r in results:
        intents = ",".join(i.intent.value for i in r.analysis.intents) or "-"
        print(f"{r.message_id[:10]:<10} {r.routing.decision.value:<18} {r.routing.priority.value:<8} "
              f"{r.confidence.score:>5.2f}  {r.routing.primary_queue[:34]:<34} {intents}", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="triage", description="Triage inbound customer messages.")
    ap.add_argument("input", help="JSON array or JSON Lines file of messages")
    ap.add_argument("--out", default="out", help="output directory (default: out/)")
    ap.add_argument("--offline", action="store_true", help="do not call the API; rules-only degraded mode")
    ap.add_argument("--model", default=None, help=f"override model (default {config.MODEL})")
    ap.add_argument("--concurrency", type=int, default=config.CONCURRENCY)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    try:
        messages, file_issues = load_messages(args.input)
    except OSError as e:
        print(f"error: cannot read {args.input}: {e}", file=sys.stderr)
        return 2

    classifier = None
    if not args.offline:
        if _has_credentials():
            from .llm import ClaudeClassifier
            classifier = ClaudeClassifier(model=args.model or config.MODEL)
        else:
            print("warning: no ANTHROPIC_API_KEY set; running in offline rules mode "
                  "(everything will go to humans). Pass --offline to silence this.", file=sys.stderr)

    pipeline = TriagePipeline(classifier, concurrency=args.concurrency)
    results = asyncio.run(pipeline.triage_many(messages))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
    summary = summarize(results, file_issues)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.quiet:
        _print_table(results)
        print(json.dumps(summary, indent=2), file=sys.stderr)
    print(f"wrote {len(results)} results to {out / 'results.jsonl'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
