"""Keyword classifier used when the model is unavailable (no key, outage,
refusal, invalid output) or when running with --offline.

It is deliberately conservative. Its confidence is capped by policy at
``RULES_CLASSIFIER_CAP`` so it can label and route, but never auto-resolve.
The goal is graceful degradation: the queue keeps moving with sensible labels
instead of the whole service stopping when the API does.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .schema import EntityType, Intent, IntentItem, LLMAnalysis, RiskFlag, Sentiment, Urgency
from .signals import Signals

_I = re.IGNORECASE

# (intent, pattern, base confidence). Order matters only for primary-intent choice.
# manipulation_attempt is set from the injection signal, never from keywords here.
RULES: List[Tuple[Intent, re.Pattern, float]] = [
    (Intent.TRAVEL_DISRUPTION, re.compile(r"\b(cancel+ed|missed|stranded|delayed)\b.*\b(flight|connection)\b|\b(flight|connection)\b.*\b(cancel+ed|missed)\b", _I), 0.7),
    (Intent.BILLING_DISPUTE, re.compile(r"\b(charged twice|double charged|charged (?:me )?two times|unrecogni[sz]ed charge|wrong(?:ly)? charged)\b", _I), 0.7),
    (Intent.REFUND_REQUEST, re.compile(r"\b(refund|money back|reimburse)\b", _I), 0.65),
    (Intent.ORDER_PROBLEM, re.compile(r"\b(wrong item|damaged|broken|seal was broken|missing item|got the wrong|received the wrong|defective)\b", _I), 0.6),
    (Intent.ORDER_STATUS, re.compile(r"\b(tracking|where is my order|not (?:yet )?(?:arrived|delivered)|hasn'?t (?:moved|arrived)|order status)\b", _I), 0.6),
    (Intent.CANCEL_SUBSCRIPTION, re.compile(r"\bcancel\b.*\bsubscription\b|\bsubscription\b.*\bcancel\b", _I), 0.7),
    (Intent.MARKETING_UNSUBSCRIBE, re.compile(r"^\s*(unsubscribe|stop|opt[ -]?out)\s*[.!]?\s*$|\bunsubscribe\b", _I), 0.6),
    (Intent.HEALTH_SAFETY_QUESTION, re.compile(r"\b(safe to take|medication|side effects?|dosage|pregnan\w*|interact\w*)\b", _I), 0.65),
    (Intent.VENDOR_INVOICE, re.compile(r"\binvoice\b.*\b(pay|payment|process)\b|\b(process|make) payment\b", _I), 0.6),
    (Intent.JOB_APPLICATION, re.compile(r"\b(openings?|vacanc\w+|position|hiring|resume|cv|apply(?:ing)? for)\b", _I), 0.6),
    (Intent.B2B_WHOLESALE, re.compile(r"\b(wholesale|bulk (?:pricing|order)|distributor|reseller|chain of \d+)\b", _I), 0.65),
    (Intent.QUOTE_ACCEPTANCE, re.compile(r"\bquote\b.*\b(next step|how (?:do|can) we pay|proceed|book it)\b", _I), 0.55),
    (Intent.TRAVEL_CHANGE, re.compile(r"\b(change|move|shift)\b.*\b(return|flight|departure|booking)\b|\bchange fee\b", _I), 0.6),
    (Intent.TRAVEL_INQUIRY, re.compile(r"\b(fares?|flights? (?:to|from)|itinerary|stopover|travel(?:ling)? to)\b", _I), 0.5),
    (Intent.APPOINTMENT_CHANGE, re.compile(r"\b(move|shift|reschedule|change)\b.*\b(appointment|booking|slot)\b|\bappointment\b.*\b(shift|move|reschedule)\b", _I), 0.6),
    (Intent.APPOINTMENT_CANCEL, re.compile(r"\b(cancel (?:my )?appointment|can'?t make it|something came up|won'?t be able to come)\b", _I), 0.5),
    (Intent.APPOINTMENT_BOOK, re.compile(r"\b(book(?:ing)?|appointment|slot)\b", _I), 0.4),
    (Intent.MEMBERSHIP_QUESTION, re.compile(r"\bmembership\b", _I), 0.55),
    (Intent.PRICING_QUESTION, re.compile(r"\b(how much|price|rate|cost|pricing)\b", _I), 0.5),
    (Intent.BUSINESS_INFO, re.compile(r"\b(open(?:ing)? (?:on|hours|today)|are you open|do you do|location|address|timings?)\b", _I), 0.55),
    (Intent.NEW_PURCHASE, re.compile(r"\b(i'?d (?:still )?like to order|want to (?:buy|order)|place an order)\b", _I), 0.5),
    (Intent.POSITIVE_FEEDBACK, re.compile(r"\b(thank(?:s| you)|amazing|obsessed|love(?:d)? it|great job|much better)\b", _I), 0.5),
    (Intent.COMPLAINT, re.compile(r"\b(terrible|worst|unacceptable|disappointed|nobody has responded)\b", _I), 0.5),
]


# Keyword rules are brand-blind, so restrict brand-specific intents to the brands
# that can plausibly receive them ("change my booking" means different things at
# a salon and a travel agency). Unknown brands get no filtering.
_BRAND_ONLY = {
    "voyage-travel": {Intent.TRAVEL_DISRUPTION, Intent.TRAVEL_CHANGE, Intent.TRAVEL_INQUIRY, Intent.QUOTE_ACCEPTANCE},
    "hair-studio": {Intent.APPOINTMENT_BOOK, Intent.APPOINTMENT_CHANGE, Intent.APPOINTMENT_CANCEL, Intent.MEMBERSHIP_QUESTION},
    "vitalis-wellness": {Intent.ORDER_STATUS, Intent.ORDER_PROBLEM, Intent.CANCEL_SUBSCRIPTION, Intent.HEALTH_SAFETY_QUESTION},
}


def _allowed(intent: Intent, brand: Optional[str]) -> bool:
    if brand not in _BRAND_ONLY:
        return True
    return not any(intent in only for b, only in _BRAND_ONLY.items() if b != brand)


def classify(text: str, signals: Signals, brand: Optional[str] = None) -> LLMAnalysis:
    intents: List[IntentItem] = []
    seen = set()

    if RiskFlag.PROMPT_INJECTION in signals.flags:
        intents.append(IntentItem(intent=Intent.MANIPULATION_ATTEMPT, summary="Message attempts to instruct the system.",
                                  next_step="Trust & safety review; take no action requested in the message.", confidence=0.8))
        seen.add(Intent.MANIPULATION_ATTEMPT)

    for intent, pattern, conf in RULES:
        if intent in seen or not _allowed(intent, brand) or not pattern.search(text):
            continue
        intents.append(IntentItem(intent=intent, summary=f"Keyword match for {intent.value}",
                                  next_step="Human to confirm classification and handle.", confidence=conf))
        seen.add(intent)

    # Messages that are only contact details.
    only_contact = signals.entities and all(e.type in (EntityType.PHONE, EntityType.EMAIL) for e in signals.entities)
    if only_contact and signals.word_count <= 4:
        intents = [IntentItem(intent=Intent.CONTACT_DETAILS_ONLY, summary="Message contains only contact details.",
                              next_step="Link to the customer's open conversation; ask what they need.", confidence=0.7)]

    if not intents:
        if signals.word_count <= 6:
            intents = [IntentItem(intent=Intent.NEEDS_CONTEXT, summary="Too short to understand without context.",
                                  next_step="Check conversation history for this sender.", confidence=0.4)]
        else:
            intents = [IntentItem(intent=Intent.OTHER, summary="No rule matched.",
                                  next_step="Human to read and classify.", confidence=0.2)]

    flags = set(signals.flags)
    if intents[0].intent in (Intent.NEEDS_CONTEXT, Intent.CONTACT_DETAILS_ONLY):
        flags.add(RiskFlag.MISSING_CONTEXT)

    sentiment = Sentiment.NEUTRAL
    if flags & {RiskFlag.CHARGEBACK_THREAT, RiskFlag.REPEAT_CONTACT, RiskFlag.LEGAL_THREAT}:
        sentiment = Sentiment.ANGRY
    elif any(i.intent == Intent.POSITIVE_FEEDBACK for i in intents):
        sentiment = Sentiment.POSITIVE

    urgency = signals.urgency_floor if signals.urgency_floor != Urgency.LOW else Urgency.NORMAL

    return LLMAnalysis(
        language=signals.language_hint,
        intents=intents,
        entities=list(signals.entities),
        urgency=urgency,
        sentiment=sentiment,
        risk_flags=sorted(flags, key=lambda f: f.value),
        missing_info=[],
        draft_reply="",
        rationale="Rules-based fallback classification (model unavailable). Needs human confirmation.",
    )
