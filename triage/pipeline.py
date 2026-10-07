"""Orchestration: normalise -> signals -> classify -> reconcile -> score -> route.

``triage_one`` never raises. Any unexpected exception for one message becomes
a HUMAN_REVIEW result for that message only; the batch carries on.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

from . import config, policy, rules
from .llm import ClaudeClassifier, LLMCallResult
from .schema import (
    Confidence, Decision, InputStatus, LLMAnalysis, NormalizedMessage, ProcessingMeta,
    Routing, TriageResult, Urgency,
)
from .signals import compute_signals

log = logging.getLogger("triage")

# Errors after which calling the API again is pointless for this run.
_FATAL_PREFIXES = ("auth_error", "bad_request")


class TriagePipeline:
    def __init__(self, classifier: Optional[ClaudeClassifier] = None, concurrency: int = config.CONCURRENCY):
        self.classifier = classifier
        self.sem = asyncio.Semaphore(max(1, concurrency))
        self.disabled_reason: Optional[str] = None if classifier else "offline mode"

    async def _call_model(self, msg: NormalizedMessage) -> LLMCallResult:
        assert self.classifier is not None
        result = await self.classifier.classify(msg)
        # One retry for outputs we couldn't use (transport errors are already retried by the SDK).
        if result.error and result.error.startswith(("invalid_output", "no_text_block", "output_truncated")):
            retry = await self.classifier.classify(msg)
            retry.cost_usd += result.cost_usd
            retry.input_tokens += result.input_tokens
            retry.output_tokens += result.output_tokens
            retry.cache_read_tokens += result.cache_read_tokens
            retry.cache_write_tokens += result.cache_write_tokens
            retry.latency_ms += result.latency_ms
            if retry.error:
                retry.error = f"{result.error}; retry: {retry.error}"
            result = retry
        if result.error and result.error.startswith(_FATAL_PREFIXES) and self.disabled_reason is None:
            self.disabled_reason = result.error
            log.error("Disabling model calls for this run: %s", result.error)
        return result

    async def triage_one(self, msg: NormalizedMessage) -> TriageResult:
        try:
            async with self.sem:
                return await self._triage(msg)
        except Exception as exc:  # last-resort isolation; one bad message must not sink the batch
            log.exception("Unexpected failure on %s", msg.id)
            return _failure_result(msg, f"internal_error: {type(exc).__name__}: {str(exc)[:200]}")

    async def _triage(self, msg: NormalizedMessage) -> TriageResult:
        start = time.monotonic()
        meta = ProcessingMeta(classifier="none", prompt_version=config.PROMPT_VERSION)
        signals = compute_signals(msg.text)

        if msg.input_status in (InputStatus.EMPTY, InputStatus.INVALID_RECORD):
            analysis = LLMAnalysis(rationale="No readable text; nothing to classify.")
            conf = policy.score_confidence(analysis, signals, msg, "none", [])
            routing = policy.route(analysis, conf, msg, "none")
            meta.latency_ms = int((time.monotonic() - start) * 1000)
            return _assemble(msg, analysis, routing, conf, meta)

        analysis: Optional[LLMAnalysis] = None
        if self.classifier and self.disabled_reason is None:
            r = await self._call_model(msg)
            meta.model = r.model
            meta.input_tokens, meta.output_tokens = r.input_tokens, r.output_tokens
            meta.cache_read_tokens, meta.cache_write_tokens = r.cache_read_tokens, r.cache_write_tokens
            meta.cost_usd = round(r.cost_usd, 6)
            if r.analysis is not None and r.analysis.intents:
                analysis, meta.classifier = r.analysis, "llm"
            elif r.analysis is not None:
                meta.errors.append("model_returned_no_intents")
            if r.error:
                meta.errors.append(r.error)
        elif self.disabled_reason:
            meta.errors.append(f"model_disabled: {self.disabled_reason}")

        if analysis is None:
            analysis, meta.classifier = rules.classify(msg.text, signals, msg.brand), "rules"

        notes = policy.merge_signals(analysis, signals, msg)
        conf = policy.score_confidence(analysis, signals, msg, meta.classifier, notes)
        routing = policy.route(analysis, conf, msg, meta.classifier)
        if notes:
            routing.reasons.append("checks: " + "; ".join(notes))
        meta.latency_ms = int((time.monotonic() - start) * 1000)
        return _assemble(msg, analysis, routing, conf, meta)

    async def triage_many(self, messages: List[NormalizedMessage]) -> List[TriageResult]:
        return list(await asyncio.gather(*(self.triage_one(m) for m in messages)))


def _assemble(msg, analysis, routing, conf, meta) -> TriageResult:
    return TriageResult(
        message_id=msg.id, brand=msg.brand, channel=msg.channel, received_at=msg.received_at,
        input_status=msg.input_status, input_issues=msg.input_issues,
        analysis=analysis, routing=routing, confidence=conf, meta=meta,
    )


def _failure_result(msg: NormalizedMessage, error: str) -> TriageResult:
    return TriageResult(
        message_id=msg.id, brand=msg.brand, channel=msg.channel, received_at=msg.received_at,
        input_status=msg.input_status, input_issues=msg.input_issues,
        analysis=LLMAnalysis(rationale="Processing failed; routed to a human."),
        routing=Routing(
            decision=Decision.HUMAN_REVIEW, primary_queue=config.TRIAGE_QUEUE, priority=Urgency.NORMAL,
            sla_minutes=config.SLA_MINUTES[Urgency.NORMAL], reasons=[error],
        ),
        confidence=Confidence(score=0.0, band="low", caps_applied=["processing_error"]),
        meta=ProcessingMeta(classifier="none", prompt_version=config.PROMPT_VERSION, errors=[error]),
    )
