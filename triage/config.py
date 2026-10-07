"""All tunable policy lives here so ops can change it without touching logic.

Anything that decides *who acts* is in this file and is deterministic. The model
never sees or chooses queue names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .schema import Intent, RiskFlag, Urgency

PROMPT_VERSION = "triage-v1.3"

KNOWN_BRANDS = {
    "vitalis-wellness": "Vitalis Wellness - supplements & wellness products, sold online, with monthly subscriptions. Ships in India (INR).",
    "hair-studio": "Hair Studio - hair salon: colour, balayage, keratin, cuts. Appointments and memberships. Clients write in English and Hinglish.",
    "voyage-travel": "Voyage Travel - travel agency: flights (many India <-> Canada routes), stopovers, quotes, booking changes.",
}
KNOWN_CHANNELS = {"whatsapp", "email", "instagram"}

MAX_TEXT_CHARS = 6000  # ~1.5k tokens. Longer bodies are truncated and flagged.

# --- Model ------------------------------------------------------------------

MODEL = os.getenv("TRIAGE_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("TRIAGE_EFFORT", "low")  # classification does not repay deep thinking
MAX_OUTPUT_TOKENS = int(os.getenv("TRIAGE_MAX_TOKENS", "4000"))
REQUEST_TIMEOUT_S = float(os.getenv("TRIAGE_TIMEOUT_S", "60"))
MAX_RETRIES = int(os.getenv("TRIAGE_MAX_RETRIES", "3"))
CONCURRENCY = int(os.getenv("TRIAGE_CONCURRENCY", "8"))
# Server-side refusal fallback is only available on the first-party Claude API.
USE_REFUSAL_FALLBACK = os.getenv("TRIAGE_REFUSAL_FALLBACK", "1") == "1"
FALLBACK_CAPABLE_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}

# USD per million tokens: (input, output, cache_read, cache_write_5m)
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-4-5": (1.00, 5.00, 0.10, 1.25),
}

# --- Decision thresholds ----------------------------------------------------

AUTO_RESOLVE_MIN_CONFIDENCE = 0.85
ROUTE_MIN_CONFIDENCE = 0.60
RULES_CLASSIFIER_CAP = 0.55  # offline/fallback classifier can never auto-resolve

SLA_MINUTES = {
    Urgency.CRITICAL: 15,
    Urgency.HIGH: 120,
    Urgency.NORMAL: 480,
    Urgency.LOW: 1440,
}


@dataclass(frozen=True)
class IntentPolicy:
    team: str  # brand-scoped unless it starts with "global."
    automatable: bool  # may automation act on this without a human?
    floor: Urgency  # minimum priority for this intent


# `automatable=True` only where the automated action is low-risk and reversible:
# a status lookup, an FAQ answer, a thank-you, a marketing opt-out (legally required
# to honour anyway). Anything that moves money, changes a booking, gives health
# advice or commits the business is a human's job.
INTENT_POLICY: dict[Intent, IntentPolicy] = {
    Intent.ORDER_STATUS: IntentPolicy("support", True, Urgency.NORMAL),
    Intent.ORDER_PROBLEM: IntentPolicy("support", False, Urgency.NORMAL),
    Intent.REFUND_REQUEST: IntentPolicy("support", False, Urgency.NORMAL),
    Intent.BILLING_DISPUTE: IntentPolicy("billing", False, Urgency.HIGH),
    Intent.CANCEL_SUBSCRIPTION: IntentPolicy("retention", False, Urgency.NORMAL),
    Intent.NEW_PURCHASE: IntentPolicy("sales", False, Urgency.NORMAL),
    Intent.PRODUCT_QUESTION: IntentPolicy("support", True, Urgency.NORMAL),
    Intent.HEALTH_SAFETY_QUESTION: IntentPolicy("product_safety", False, Urgency.HIGH),
    Intent.PRICING_QUESTION: IntentPolicy("front_desk", True, Urgency.NORMAL),
    Intent.BUSINESS_INFO: IntentPolicy("front_desk", True, Urgency.LOW),
    Intent.APPOINTMENT_BOOK: IntentPolicy("front_desk", False, Urgency.NORMAL),
    Intent.APPOINTMENT_CHANGE: IntentPolicy("front_desk", False, Urgency.NORMAL),
    Intent.APPOINTMENT_CANCEL: IntentPolicy("front_desk", False, Urgency.NORMAL),
    Intent.MEMBERSHIP_QUESTION: IntentPolicy("front_desk", True, Urgency.NORMAL),
    Intent.TRAVEL_INQUIRY: IntentPolicy("sales", False, Urgency.NORMAL),
    Intent.TRAVEL_CHANGE: IntentPolicy("reservations", False, Urgency.NORMAL),
    Intent.TRAVEL_DISRUPTION: IntentPolicy("emergency_desk", False, Urgency.CRITICAL),
    Intent.QUOTE_ACCEPTANCE: IntentPolicy("sales", False, Urgency.HIGH),
    Intent.B2B_WHOLESALE: IntentPolicy("b2b_sales", False, Urgency.NORMAL),
    Intent.JOB_APPLICATION: IntentPolicy("global.recruiting", False, Urgency.LOW),
    Intent.VENDOR_INVOICE: IntentPolicy("global.accounts_payable", False, Urgency.NORMAL),
    Intent.MARKETING_UNSUBSCRIBE: IntentPolicy("global.marketing_ops", True, Urgency.LOW),
    Intent.POSITIVE_FEEDBACK: IntentPolicy("community", True, Urgency.LOW),
    Intent.COMPLAINT: IntentPolicy("support", False, Urgency.HIGH),
    Intent.CONTACT_DETAILS_ONLY: IntentPolicy("support", False, Urgency.NORMAL),
    Intent.NEEDS_CONTEXT: IntentPolicy("support", False, Urgency.NORMAL),
    Intent.MANIPULATION_ATTEMPT: IntentPolicy("global.trust_safety", False, Urgency.HIGH),
    Intent.OTHER: IntentPolicy("support", False, Urgency.NORMAL),
}

# Any of these flags means a human sees the message, regardless of confidence.
HUMAN_REQUIRED_FLAGS = {
    RiskFlag.PROMPT_INJECTION,
    RiskFlag.CHARGEBACK_THREAT,
    RiskFlag.LEGAL_THREAT,
    RiskFlag.REPEAT_CONTACT,
    RiskFlag.HEALTH_SAFETY,
    RiskFlag.VULNERABLE_CUSTOMER,
    RiskFlag.TIME_CRITICAL,
    RiskFlag.PAYMENT_FRAUD_RISK,
    RiskFlag.MISSING_ATTACHMENT,
    RiskFlag.MISSING_CONTEXT,
    RiskFlag.MONEY_MOVEMENT,
    RiskFlag.AMBIGUOUS_REQUEST,
}

# Flags that raise priority to at least this level.
FLAG_PRIORITY_FLOOR = {
    RiskFlag.CHARGEBACK_THREAT: Urgency.HIGH,
    RiskFlag.LEGAL_THREAT: Urgency.HIGH,
    RiskFlag.REPEAT_CONTACT: Urgency.HIGH,
    RiskFlag.HEALTH_SAFETY: Urgency.HIGH,
    RiskFlag.PROMPT_INJECTION: Urgency.HIGH,
    RiskFlag.TIME_CRITICAL: Urgency.CRITICAL,
}

# Flags that send the whole message to a specific queue first.
FLAG_QUEUE_OVERRIDE = {
    RiskFlag.PROMPT_INJECTION: "global.trust_safety",
    RiskFlag.PAYMENT_FRAUD_RISK: "global.accounts_payable",
}

TRIAGE_QUEUE = "global.triage"  # where uncertain messages land for a human look


def queue_for(team: str, brand: str | None) -> str:
    if team.startswith("global."):
        return team
    return f"{brand if brand in KNOWN_BRANDS else 'unknown-brand'}.{team}"
