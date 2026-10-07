"""Deterministic signals: regex entity extraction and risk detection.

These run on every message, before and independently of the model, for three
reasons:
1. Safety-critical flags (prompt injection, chargeback threats, fraud patterns,
   stranded travellers) must not depend on a model that the message itself may
   be trying to manipulate.
2. They give an independent check on the model's extraction (did it miss an
   order id? did it invent one?), which feeds the confidence score.
3. They power the offline / degraded-mode classifier when the API is down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

from .schema import Entity, EntityType, RiskFlag, Urgency

_I = re.IGNORECASE

ORDER_ID = re.compile(r"\b[A-Z]{2,4}-\d{4,}\b")
BOOKING_REF_CTX = re.compile(r"\b(?:booking\s*ref(?:erence)?|PNR|confirmation|ref)\s*(?:no\.?|number|#|:)?\s*([A-Z0-9]{5,8})\b", _I)
ALNUM_CODE = re.compile(r"\b(?=[A-Z0-9]*\d)(?=(?:[A-Z0-9]*[A-Z]){2})[A-Z0-9]{6,8}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE_CANDIDATE = re.compile(r"(?<![\w-])\+?\d[\d\s\-().]{7,}\d(?![\w-])")
AMOUNT = re.compile(
    r"(?:(?P<cur1>INR|Rs\.?|₹|USD|CAD|AED|US\$|C\$|\$)\s?(?P<num1>\d[\d,]*(?:\.\d+)?))"
    r"|(?:(?P<num2>\d[\d,]*(?:\.\d+)?)\s?(?P<cur2>INR|rupees|USD|CAD|AED|dollars))",
    _I,
)

INJECTION_PATTERNS = [
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above|earlier) (?:instructions|prompts|rules)",
    r"disregard (?:all |any )?(?:previous|prior|above|your) (?:instructions|rules|guidelines)",
    r"\byou are now\b",
    r"\b(?:administrator|admin|developer|god|jailbreak|dan) mode\b",
    r"\bsystem (?:notice|prompt|override|message)\s*:",
    r"\bdo not (?:escalate|forward|show) (?:this|to)\b",
    r"\b(?:reply|respond) (?:only )?with ['\"]?confirmed\b",
    r"\bnew instructions\b",
    r"\bact as\b.*\b(?:assistant|ai|model|system)\b",
    r"</?(?:system|instructions?|customer_message)>",
]
INJECTION = re.compile("|".join(INJECTION_PATTERNS), _I)

CHARGEBACK = re.compile(r"\b(?:chargeback|charge back|raising it with my bank|dispute (?:it |this )?with (?:my|the) (?:bank|card)|report (?:it |this )?to (?:my|the) bank|credit card company)\b", _I)
LEGAL = re.compile(r"\b(?:lawyer|legal action|consumer (?:court|forum)|sue you|lawsuit|legal notice)\b", _I)
REPEAT = re.compile(r"\b(?:(?:second|third|fourth|fifth|2nd|3rd|4th|5th|another) time (?:i'?m |i am )?(?:writing|asking|contacting|messaging)|nobody (?:has )?(?:responded|replied|got back)|no one (?:has )?(?:responded|replied)|still (?:no|haven'?t had a) (?:response|reply)|again and again)\b", _I)
HEALTH = re.compile(r"\b(?:safe to take|medication|medicine|interact(?:ion)?s?|side effects?|dosage|dose|pregnan\w*|breastfeed\w*|allerg\w*|blood pressure|diabet\w*|prescri\w*|my doctor)\b", _I)
VULNERABLE = re.compile(r"\b(?:elderly|wheelchair|disab\w*|over (?:70|75|80)|senior citizen|infant|newborn|pregnan\w*|mobility)\b", _I)
STRANDED = re.compile(r"\b(?:stranded|stuck at|missed (?:my |our |the )?(?:flight|connection)|(?:flight|connection) (?:\w+ ){0,3}(?:was |is |got )?cancel+ed|next available is)\b", _I)
AIRPORT_NOW = re.compile(r"\b(?:we are|i am|i'm|we're) at (?:the )?(?:airport|pearson|terminal|[A-Z]{3}\b)", _I)
URGENT_WORD = re.compile(r"\b(?:urgent|urgently|immediately|asap|emergency)\b", _I)
PAYMENT_FRAUD = re.compile(r"\b(?:bank details|bank account (?:details|number)|iban|swift|wire transfer|remit(?:tance)?|process (?:the )?payment|new account number|changed? (?:our|my) bank)\b", _I)
INVOICE = re.compile(r"\binvoice\b", _I)
ATTACHMENT_REF = re.compile(r"\b(?:attached|attachment|enclosed|see (?:the )?(?:pdf|file|screenshot)|resume attached|cv attached)\b", _I)
MONEY = re.compile(r"\b(?:refund|money back|charged twice|double charged|reimburse)\b", _I)
HINGLISH = re.compile(r"\b(?:meri|mera|mujhe|aap|kya|hai|thi|tha|aur|bhi|bata|dena|karna|baje|kal|ki|ka|ke|nahi|kyun)\b", _I)
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


@dataclass
class Signals:
    entities: List[Entity] = field(default_factory=list)
    flags: Set[RiskFlag] = field(default_factory=set)
    urgency_floor: Urgency = Urgency.LOW
    language_hint: str = "unknown"
    word_count: int = 0
    notes: List[str] = field(default_factory=list)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def extract_entities(text: str) -> List[Entity]:
    out: List[Entity] = []
    seen: Set[tuple] = set()

    def add(t: EntityType, value: str, normalized: str = "") -> None:
        key = (t, normalized or value)
        if key not in seen:
            seen.add(key)
            out.append(Entity(type=t, value=value, normalized=normalized or value, grounded=True, source="regex"))

    order_ids = set(ORDER_ID.findall(text))
    for oid in order_ids:
        add(EntityType.ORDER_ID, oid)
    for m in BOOKING_REF_CTX.finditer(text):
        code = m.group(1)
        if any(c.isdigit() for c in code) and code.upper() == code:
            add(EntityType.BOOKING_REF, code)
    for code in ALNUM_CODE.findall(text):
        if code not in order_ids:
            add(EntityType.BOOKING_REF, code)
    for e in EMAIL.findall(text):
        add(EntityType.EMAIL, e, e.lower())
    for m in PHONE_CANDIDATE.finditer(text):
        raw = m.group(0).strip()
        d = _digits(raw)
        if 10 <= len(d) <= 13 and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            norm = ("+" + d) if raw.startswith("+") else d
            add(EntityType.PHONE, raw, norm)
    for m in AMOUNT.finditer(text):
        cur = (m.group("cur1") or m.group("cur2") or "").upper().rstrip(".")
        cur = {"RS": "INR", "₹": "INR", "RUPEES": "INR", "$": "USD?", "US$": "USD", "C$": "CAD", "DOLLARS": "USD?"}.get(cur, cur)
        num = (m.group("num1") or m.group("num2") or "").replace(",", "")
        add(EntityType.AMOUNT, m.group(0).strip(), f"{cur} {num}")
    return out


def compute_signals(text: str) -> Signals:
    s = Signals()
    s.word_count = len(text.split())
    if not text:
        return s
    s.entities = extract_entities(text)

    def flag(f: RiskFlag, note: str) -> None:
        s.flags.add(f)
        s.notes.append(note)

    if INJECTION.search(text):
        flag(RiskFlag.PROMPT_INJECTION, "injection pattern matched")
    if CHARGEBACK.search(text):
        flag(RiskFlag.CHARGEBACK_THREAT, "chargeback/bank escalation language")
    if LEGAL.search(text):
        flag(RiskFlag.LEGAL_THREAT, "legal threat language")
    if REPEAT.search(text):
        flag(RiskFlag.REPEAT_CONTACT, "customer says they have written before without reply")
    if HEALTH.search(text):
        flag(RiskFlag.HEALTH_SAFETY, "health/medication language")
    if VULNERABLE.search(text):
        flag(RiskFlag.VULNERABLE_CUSTOMER, "vulnerable person mentioned")
    if STRANDED.search(text) or (AIRPORT_NOW.search(text) and URGENT_WORD.search(text)):
        flag(RiskFlag.TIME_CRITICAL, "traveller disruption in progress")
    if PAYMENT_FRAUD.search(text) or (INVOICE.search(text) and re.search(r"\bpay(?:ment)?\b", text, _I)):
        flag(RiskFlag.PAYMENT_FRAUD_RISK, "unsolicited payment / bank-detail request")
    if ATTACHMENT_REF.search(text):
        flag(RiskFlag.MISSING_ATTACHMENT, "text references an attachment; the record has none")
    if MONEY.search(text):
        flag(RiskFlag.MONEY_MOVEMENT, "request would move money")
    if any(e.type in (EntityType.PHONE, EntityType.EMAIL) for e in s.entities):
        s.flags.add(RiskFlag.PII_PRESENT)

    if RiskFlag.TIME_CRITICAL in s.flags:
        s.urgency_floor = Urgency.CRITICAL
    elif URGENT_WORD.search(text) or s.flags & {RiskFlag.CHARGEBACK_THREAT, RiskFlag.REPEAT_CONTACT, RiskFlag.LEGAL_THREAT}:
        s.urgency_floor = Urgency.HIGH

    if DEVANAGARI.search(text):
        s.language_hint = "hi"
    elif len(HINGLISH.findall(text)) >= 2:
        s.language_hint = "hi-Latn+en"
    elif re.search(r"[A-Za-z]", text):
        s.language_hint = "en"
    return s


def normalize_for_match(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def is_grounded(entity: Entity, text: str) -> bool:
    """Does the entity's verbatim value actually occur in the message?

    Dates/times are allowed to be relative ("tomorrow", "4 baje"), so for those
    we only require *some* verbatim value. For identifiers and numbers we
    require the characters to be present, which catches invented order ids.
    """
    value = (entity.value or "").strip()
    if not value:
        return False
    hay = normalize_for_match(text)
    needle = normalize_for_match(value)
    if needle and needle in hay:
        return True
    if entity.type == EntityType.PHONE:
        return _digits(value) != "" and _digits(value) in _digits(text)
    return value.lower() in text.lower()
