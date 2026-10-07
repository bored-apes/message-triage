"""Output contract for a triage result.

Design notes
------------
* A message can carry several requests ("refund this order, and also send me two
  more tubs"), so ``intents`` is a list and every intent gets its own task/queue.
  A single ``category`` field would not survive MSG-006 / MSG-008 / MSG-014.
* What the model produces (``LLMAnalysis``) is kept separate from what the
  service decides (``Routing``, ``Confidence``). The model proposes; deterministic
  code disposes. Nothing the customer writes can pick the queue or the decision.
* Every list defaults to empty and every optional field to ``None`` so that a
  result can always be built, even for a null or garbage input.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    # Commerce / orders
    ORDER_STATUS = "order_status"
    ORDER_PROBLEM = "order_problem"  # damaged, wrong item, missing item
    REFUND_REQUEST = "refund_request"
    BILLING_DISPUTE = "billing_dispute"  # double charge, unrecognised charge
    CANCEL_SUBSCRIPTION = "cancel_subscription"
    NEW_PURCHASE = "new_purchase"
    PRODUCT_QUESTION = "product_question"
    HEALTH_SAFETY_QUESTION = "health_safety_question"  # interactions, dosage, side effects
    # Services / appointments
    PRICING_QUESTION = "pricing_question"
    BUSINESS_INFO = "business_info"  # hours, location, services offered
    APPOINTMENT_BOOK = "appointment_book"
    APPOINTMENT_CHANGE = "appointment_change"
    APPOINTMENT_CANCEL = "appointment_cancel"
    MEMBERSHIP_QUESTION = "membership_question"
    # Travel
    TRAVEL_INQUIRY = "travel_inquiry"  # new trip, fares, quote request
    TRAVEL_CHANGE = "travel_change"  # change an existing booking
    TRAVEL_DISRUPTION = "travel_disruption"  # cancelled/missed flight, stranded
    QUOTE_ACCEPTANCE = "quote_acceptance"  # wants to proceed / pay on a quote
    # Not customer service
    B2B_WHOLESALE = "b2b_wholesale"
    JOB_APPLICATION = "job_application"
    VENDOR_INVOICE = "vendor_invoice"
    MARKETING_UNSUBSCRIBE = "marketing_unsubscribe"
    # Sentiment-only / low-content
    POSITIVE_FEEDBACK = "positive_feedback"
    COMPLAINT = "complaint"
    CONTACT_DETAILS_ONLY = "contact_details_only"
    NEEDS_CONTEXT = "needs_context"  # "what about the other one"
    MANIPULATION_ATTEMPT = "manipulation_attempt"  # prompt injection / social engineering
    OTHER = "other"


class EntityType(str, Enum):
    ORDER_ID = "order_id"
    BOOKING_REF = "booking_ref"
    PHONE = "phone"
    EMAIL = "email"
    AMOUNT = "amount"
    DATE = "date"
    TIME = "time"
    PRODUCT = "product"
    SERVICE = "service"
    PERSON = "person"  # staff names, travellers
    LOCATION = "location"
    CARRIER = "carrier"
    QUANTITY = "quantity"
    ORGANIZATION = "organization"
    OTHER = "other"


class RiskFlag(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    CHARGEBACK_THREAT = "chargeback_threat"
    LEGAL_THREAT = "legal_threat"
    REPEAT_CONTACT = "repeat_contact"  # "third time I'm writing"
    HEALTH_SAFETY = "health_safety"
    VULNERABLE_CUSTOMER = "vulnerable_customer"  # elderly, disability, medical need
    TIME_CRITICAL = "time_critical"  # stranded, same-day
    PAYMENT_FRAUD_RISK = "payment_fraud_risk"  # invoices with bank details, pay-here requests
    MISSING_ATTACHMENT = "missing_attachment"  # "attached" but nothing attached
    AMBIGUOUS_REQUEST = "ambiguous_request"
    MISSING_CONTEXT = "missing_context"
    MONEY_MOVEMENT = "money_movement"  # refund / payment would move money
    PII_PRESENT = "pii_present"


class Urgency(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


class Decision(str, Enum):
    AUTO_RESOLVE = "auto_resolve"  # automation acts (sends reply / performs safe action)
    ROUTE_TO_TEAM = "route_to_team"  # a human team does the work; automation only filed it
    HUMAN_REVIEW = "human_review"  # triage itself is unreliable; a human must look first
    URGENT_ESCALATION = "urgent_escalation"  # page a human now


class InputStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"  # null / whitespace / no text at all
    COERCED = "coerced"  # text was not a string; converted
    TRUNCATED = "truncated"
    INVALID_RECORD = "invalid_record"  # not an object / unrecoverable


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class NormalizedMessage(BaseModel):
    id: str
    brand: Optional[str] = None
    channel: Optional[str] = None
    received_at: Optional[str] = None  # ISO-8601 as received, if parseable
    text: str = ""
    attachment_count: int = 0
    input_status: InputStatus = InputStatus.OK
    input_issues: List[str] = Field(default_factory=list)
    content_hash: str = ""
    source_index: Optional[int] = None


# ---------------------------------------------------------------------------
# What the classifier (LLM or rules) produces
# ---------------------------------------------------------------------------


class Entity(BaseModel):
    type: EntityType
    value: str  # verbatim span from the message
    normalized: str = ""  # e.g. ISO date, E.164 phone, "INR 3780"
    grounded: Optional[bool] = None  # set by the service: does `value` appear in the text?
    source: str = "model"  # model | regex


class IntentItem(BaseModel):
    intent: Intent
    summary: str = ""
    next_step: str = ""
    confidence: float = 0.0


class LLMAnalysis(BaseModel):
    language: str = "unknown"
    intents: List[IntentItem] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    urgency: Urgency = Urgency.NORMAL
    sentiment: Sentiment = Sentiment.NEUTRAL
    risk_flags: List[RiskFlag] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    draft_reply: str = ""
    rationale: str = ""


# ---------------------------------------------------------------------------
# What the service decides
# ---------------------------------------------------------------------------


class Task(BaseModel):
    intent: Intent
    queue: str
    automatable: bool
    summary: str = ""
    next_step: str = ""


class Routing(BaseModel):
    decision: Decision
    primary_queue: str
    priority: Urgency
    sla_minutes: int
    reasons: List[str] = Field(default_factory=list)
    tasks: List[Task] = Field(default_factory=list)
    send_draft_reply: bool = False  # only true when decision == auto_resolve


class Confidence(BaseModel):
    score: float
    band: str  # high | medium | low
    components: dict = Field(default_factory=dict)
    caps_applied: List[str] = Field(default_factory=list)


class ProcessingMeta(BaseModel):
    classifier: str  # llm | rules | none
    model: Optional[str] = None
    prompt_version: str = ""
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    errors: List[str] = Field(default_factory=list)


class TriageResult(BaseModel):
    message_id: str
    brand: Optional[str] = None
    channel: Optional[str] = None
    received_at: Optional[str] = None
    input_status: InputStatus
    input_issues: List[str] = Field(default_factory=list)
    analysis: LLMAnalysis
    routing: Routing
    confidence: Confidence
    meta: ProcessingMeta
