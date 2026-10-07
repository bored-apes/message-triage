"""HTTP service.

    uvicorn triage.api:app --port 8000

POST /triage        one message object  -> TriageResult
POST /triage/batch  list of messages    -> {"results": [...], "file_issues": [...]}
GET  /healthz
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import config
from .ingest import normalize_record, parse_messages
from .pipeline import TriagePipeline

MAX_BODY_BYTES = 5 * 1024 * 1024

app = FastAPI(title="Message triage", version=config.PROMPT_VERSION)
_pipeline: TriagePipeline | None = None


def pipeline() -> TriagePipeline:
    global _pipeline
    if _pipeline is None:
        classifier = None
        if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
            from .llm import ClaudeClassifier
            classifier = ClaudeClassifier()
        _pipeline = TriagePipeline(classifier)
    return _pipeline


async def _body(request: Request) -> tuple[str | None, JSONResponse | None]:
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        return None, JSONResponse({"error": "payload too large"}, status_code=413)
    return raw.decode("utf-8", errors="replace"), None


@app.get("/healthz")
async def healthz() -> dict:
    p = pipeline()
    return {
        "status": "ok",
        "model": config.MODEL if p.classifier else None,
        "mode": "llm" if p.classifier and not p.disabled_reason else "rules",
        "disabled_reason": p.disabled_reason,
        "prompt_version": config.PROMPT_VERSION,
    }


@app.post("/triage")
async def triage_single(request: Request):
    text, err = await _body(request)
    if err:
        return err
    # Accept anything; a broken body becomes a routed-to-human result, not a 500.
    try:
        payload: Any = json.loads(text) if text and text.strip() else None
    except json.JSONDecodeError:
        payload = {"text": text, "id": None}
    msg = normalize_record(payload, 0)
    result = await pipeline().triage_one(msg)
    return JSONResponse(json.loads(result.model_dump_json()))


@app.post("/triage/batch")
async def triage_batch(request: Request):
    text, err = await _body(request)
    if err:
        return err
    messages, issues = parse_messages(text or "")
    results = await pipeline().triage_many(messages)
    return JSONResponse({"results": [json.loads(r.model_dump_json()) for r in results], "file_issues": issues})
