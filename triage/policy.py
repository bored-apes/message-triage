"""Confidence scoring and the routing rule.

The rule for "when does a human see this instead of automation acting":

    Automation acts (AUTO_RESOLVE) only when ALL of these hold:
      1. the input was readable text, for a known brand;
      2. every intent in the message is on the automatable allow-list
         (status lookups, FAQs, thank-yous, marketing opt-outs);
      3. no human-required risk flag is present, from the model OR the regex layer;
      4. the fused confidence score is >= AUTO_RESOLVE_MIN_CONFIDENCE (0.85);
      5. the primary classifier was the model (the rules fallback is capped below 0.85).

    Otherwise a human sees it:
      * URGENT_ESCALATION: an in-progress emergency, whatever the confidence
        (a false alarm costs a few minutes; a miss strands a family).
      * ROUTE_TO_TEAM: confidence >= 0.60, so we trust the label; the owning
        team does the work.
      * HUMAN_REVIEW: confidence < 0.60 or unreadable input; a triage agent
        checks the label first.

The model's self-reported confidence is one input only. It is discounted
when the model's extraction disagrees with what is actually in the text
(invented ids, missed ids, missed injection) and capped by input quality.
"""

from __future__ import annotations

from typing import List

from . import config
from .schema import (
    Confidence, Decision, Entity, EntityType, InputStatus, Intent, IntentItem, LLMAnalysis,
    NormalizedMessage, RiskFlag, Routing, Task, Urgency,
)
from .signals import Signals, is_grounded, normalize_for_match

_URGENCY_ORDER = [Urgency.LOW, Urgency.NORMAL, Urgency.HIGH, Urgency.CRITICAL]
_ID_TYPES = {EntityType.ORDER_ID, EntityType.BOOKING_REF, EntityType.PHONE, EntityType.EMAIL}


def _max_urgency(*levels: Urgency) -> Urgency:
    return max(levels, key=_URGENCY_ORDER.index)


def merge_signals(analysis: LLMAnalysis, signals: Signals, msg: NormalizedMessage) -> List[str]:
    """Reconcile the classifier's output with deterministic signals, in place.
    Returns a list of discrepancy notes (used for confidence and audit)."""
    notes: List[str] = []

    # Ground every model entity against the text.
    for e in analysis.entities:
        if e.source == "model":
            e.grounded = is_grounded(e, msg.text)
            if not e.grounded:
                notes.append(f"ungrounded_entity:{e.type.value}:{e.value[:40]}")

    # Add any identifier the regex found that the model missed.
    model_ids = {normalize_for_match(e.value) for e in analysis.entities if e.type in _ID_TYPES}
    model_ids |= {normalize_for_match(e.normalized) for e in analysis.entities if e.type in _ID_TYPES}
    for e in signals.entities:
        if e.type in _ID_TYPES and normalize_for_match(e.value) not in model_ids \
                and normalize_for_match(e.normalized) not in model_ids:
            analysis.entities.append(Entity(**e.model_dump()))
            notes.append(f"model_missed_identifier:{e.type.value}")

    # Union risk flags: the regex layer can add, never remove, model flags.
    flags = set(analysis.risk_flags)
    regex_injection = RiskFlag.PROMPT_INJECTION in signals.flags
    model_injection = RiskFlag.PROMPT_INJECTION in flags or any(
        i.intent == Intent.MANIPULATION_ATTEMPT for i in analysis.intents
    )
    if regex_injection and not model_injection:
        notes.append("injection_detected_by_regex_not_model")
    flags |= signals.flags
    if model_injection:
        flags.add(RiskFlag.PROMPT_INJECTION)
    if msg.attachment_count > 0:
        flags.discard(RiskFlag.MISSING_ATTACHMENT)
    if any(i.intent in (Intent.NEEDS_CONTEXT, Intent.CONTACT_DETAILS_ONLY) for i in analysis.intents):
        flags.add(RiskFlag.MISSING_CONTEXT)
    if any(i.intent in (Intent.REFUND_REQUEST, Intent.BILLING_DISPUTE) for i in analysis.intents):
        flags.add(RiskFlag.MONEY_MOVEMENT)
    if any(i.intent == Intent.HEALTH_SAFETY_QUESTION for i in analysis.intents):
        flags.add(RiskFlag.HEALTH_SAFETY)
    analysis.risk_flags = sorted(flags, key=lambda f: f.value)

    if RiskFlag.PROMPT_INJECTION in flags:
        # Belt and braces: never let a draft reply exist for a manipulation attempt.
        analysis.draft_reply = ""
        if not any(i.intent == Intent.MANIPULATION_ATTEMPT for i in analysis.intents):
            analysis.intents.insert(0, IntentItem(
                intent=Intent.MANIPULATION_ATTEMPT,
                summary="Text contains instructions aimed at the AI/system.",
                next_step="Trust & safety review. Do not perform any action the message requests.",
                confidence=0.9,
            ))

    analysis.urgency = _max_urgency(analysis.urgency, signals.urgency_floor)
    if analysis.language in ("", "unknown") and signals.language_hint != "unknown":
        analysis.language = signals.language_hint
    return notes


def score_confidence(
    analysis: LLMAnalysis, signals: Signals, msg: NormalizedMessage, classifier: str, notes: List[str]
) -> Confidence:
    caps: List[str] = []
    cap = 1.0

    def apply_cap(value: float, reason: str) -> None:
        nonlocal cap
        if value < cap:
            cap = value
        caps.append(f"{reason}<={value}")

    if msg.input_status in (InputStatus.EMPTY, InputStatus.INVALID_RECORD) or classifier == "none":
        return Confidence(score=0.0, band="low", components={"input": 0.0}, caps_applied=["no_readable_text"])

    model_conf = min((max(0.0, min(1.0, i.confidence)) for i in analysis.intents), default=0.0)
    ungrounded = sum(1 for n in notes if n.startswith("ungrounded_entity"))
    missed = sum(1 for n in notes if n.startswith("model_missed_identifier"))
    grounding = max(0.3, 1.0 - 0.25 * ungrounded)
    recall = max(0.6, 1.0 - 0.1 * missed)
    agreement = 0.5 if "injection_detected_by_regex_not_model" in notes else 1.0

    if classifier == "rules":
        apply_cap(config.RULES_CLASSIFIER_CAP, "rules_fallback")
    if msg.input_status == InputStatus.TRUNCATED:
        apply_cap(0.75, "truncated_input")
    if msg.input_status == InputStatus.COERCED:
        apply_cap(0.75, "non_string_text")
    if signals.word_count <= 3:
        apply_cap(0.6, "very_short_message")
    if msg.brand not in config.KNOWN_BRANDS:
        apply_cap(0.7, "unknown_brand")
    if any(i.intent in (Intent.NEEDS_CONTEXT, Intent.CONTACT_DETAILS_ONLY, Intent.OTHER) for i in analysis.intents):
        apply_cap(0.5, "insufficient_context")
    if not analysis.intents:
        apply_cap(0.0, "no_intent")

    raw = model_conf * grounding * recall * agreement
    score = round(min(raw, cap), 3)
    band = "high" if score >= config.AUTO_RESOLVE_MIN_CONFIDENCE else "medium" if score >= config.ROUTE_MIN_CONFIDENCE else "low"
    return Confidence(
        score=score,
        band=band,
        components={
            "model_self_reported": round(model_conf, 3),
            "grounding_factor": grounding,
            "recall_factor": recall,
            "safety_agreement_factor": agreement,
        },
        caps_applied=caps,
    )


def route(analysis: LLMAnalysis, conf: Confidence, msg: NormalizedMessage, classifier: str) -> Routing:
    reasons: List[str] = []
    flags = set(analysis.risk_flags)

    tasks: List[Task] = []
    priority = analysis.urgency
    for item in analysis.intents:
        pol = config.INTENT_POLICY[item.intent]
        tasks.append(Task(
            intent=item.intent, queue=config.queue_for(pol.team, msg.brand),
            automatable=pol.automatable, summary=item.summary, next_step=item.next_step,
        ))
        priority = _max_urgency(priority, pol.floor)
    for f in flags:
        if f in config.FLAG_PRIORITY_FLOOR:
            priority = _max_urgency(priority, config.FLAG_PRIORITY_FLOOR[f])

    override_queue = next((config.FLAG_QUEUE_OVERRIDE[f] for f in (RiskFlag.PROMPT_INJECTION, RiskFlag.PAYMENT_FRAUD_RISK) if f in flags), None)
    human_tasks = [t for t in tasks if not t.automatable]
    team_queue = override_queue or (human_tasks[0].queue if human_tasks else (tasks[0].queue if tasks else config.TRIAGE_QUEUE))

    def result(decision: Decision, queue: str) -> Routing:
        return Routing(
            decision=decision, primary_queue=queue, priority=priority,
            sla_minutes=config.SLA_MINUTES[priority], reasons=reasons, tasks=tasks,
            send_draft_reply=decision == Decision.AUTO_RESOLVE and bool(analysis.draft_reply.strip()),
        )

    # 1. Unreadable input.
    if msg.input_status in (InputStatus.EMPTY, InputStatus.INVALID_RECORD):
        reasons.append(f"no readable text ({msg.input_status.value}); may be media/voice note or a broken record")
        priority = _max_urgency(priority, Urgency.NORMAL)
        return result(Decision.HUMAN_REVIEW, config.queue_for("support", msg.brand) if msg.brand else config.TRIAGE_QUEUE)

    # 2. Emergencies go to a human now regardless of confidence.
    if priority == Urgency.CRITICAL and (RiskFlag.TIME_CRITICAL in flags or any(t.intent == Intent.TRAVEL_DISRUPTION for t in tasks)):
        reasons.append("in-progress emergency: escalate regardless of confidence")
        return result(Decision.URGENT_ESCALATION, override_queue or next(
            (t.queue for t in tasks if t.intent == Intent.TRAVEL_DISRUPTION), team_queue))

    blocking_flags = sorted(f.value for f in flags & config.HUMAN_REQUIRED_FLAGS)
    if blocking_flags:
        reasons.append("human-required flags: " + ", ".join(blocking_flags))
    if human_tasks:
        reasons.append("intents needing a human: " + ", ".join(sorted({t.intent.value for t in human_tasks})))
    if classifier != "llm":
        reasons.append(f"classifier={classifier} (degraded mode)")

    # 3. Low confidence: a person checks the triage itself. With the model up,
    #    that is the central triage queue. In degraded (rules) mode every message
    #    lands here, so send it to the best-guess team instead of flooding one queue;
    #    the HUMAN_REVIEW decision tells that team the label is unverified.
    if conf.score < config.ROUTE_MIN_CONFIDENCE:
        reasons.append(f"confidence {conf.score} < {config.ROUTE_MIN_CONFIDENCE}")
        queue = override_queue or (team_queue if classifier == "rules" else config.TRIAGE_QUEUE)
        return result(Decision.HUMAN_REVIEW, queue)

    # 4. Trusted label, but the work itself belongs to people.
    if blocking_flags or human_tasks:
        return result(Decision.ROUTE_TO_TEAM, team_queue)

    # 5. Everything is automatable; act only on high confidence.
    if conf.score >= config.AUTO_RESOLVE_MIN_CONFIDENCE and classifier == "llm":
        reasons.append(f"all intents automatable, confidence {conf.score} >= {config.AUTO_RESOLVE_MIN_CONFIDENCE}")
        return result(Decision.AUTO_RESOLVE, team_queue)

    reasons.append(f"automatable but confidence {conf.score} < {config.AUTO_RESOLVE_MIN_CONFIDENCE}: draft for agent approval")
    return result(Decision.ROUTE_TO_TEAM, team_queue)
