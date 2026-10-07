"""Loading and normalising raw input.

Contract: ``load_messages`` never raises on bad *content*. Whatever comes in
(invalid JSON, JSONL, a dict instead of a list, null text, numbers, nested
objects, duplicate ids, missing ids, invisible characters, megabyte-long
bodies) is turned into a list of ``NormalizedMessage`` plus a list of
file-level issues. Only an unreadable path raises (that is an operator error,
not a data error).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Tuple

from .config import KNOWN_BRANDS, KNOWN_CHANNELS, MAX_TEXT_CHARS
from .schema import InputStatus, NormalizedMessage

# Zero-width and bidi-control characters are a classic way to hide instructions
# from a human reviewer while the model still reads them. Strip them.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤⁦-⁩﻿­]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_TEXT_KEYS = ("text", "body", "message", "content", "caption")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def _coerce_text(raw: Any, issues: List[str]) -> Tuple[str, InputStatus]:
    if raw is None:
        issues.append("text_null")
        return "", InputStatus.EMPTY
    if isinstance(raw, str):
        return raw, InputStatus.OK
    if isinstance(raw, bool) or isinstance(raw, (int, float)):
        issues.append(f"text_was_{type(raw).__name__}")
        return str(raw), InputStatus.COERCED
    if isinstance(raw, list):
        issues.append("text_was_list")
        parts = [p if isinstance(p, str) else json.dumps(p, ensure_ascii=False, default=str) for p in raw]
        return "\n".join(parts), InputStatus.COERCED
    if isinstance(raw, dict):
        issues.append("text_was_object")
        for key in _TEXT_KEYS:
            if isinstance(raw.get(key), str):
                return raw[key], InputStatus.COERCED
        return json.dumps(raw, ensure_ascii=False, default=str), InputStatus.COERCED
    issues.append(f"text_unsupported_type_{type(raw).__name__}")
    return str(raw), InputStatus.COERCED


def _clean_label(value: Any) -> str | None:
    if value is None:
        return None
    s = clean_text(str(value)).lower()
    return s[:64] or None


def _parse_timestamp(value: Any, issues: List[str]) -> str | None:
    if value is None or value == "":
        issues.append("received_at_missing")
        return None
    try:
        s = str(value).strip()
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        issues.append("received_at_unparseable")
        return None
    if dt.tzinfo is None:
        issues.append("received_at_no_timezone")
    return dt.isoformat()


def normalize_record(raw: Any, index: int) -> NormalizedMessage:
    issues: List[str] = []

    if isinstance(raw, str):
        # A bare string in the list is almost certainly the message body itself.
        issues.append("record_was_bare_string")
        raw = {"text": raw}

    if not isinstance(raw, dict):
        issues.append(f"record_not_object_{type(raw).__name__}")
        return NormalizedMessage(
            id=f"INVALID-{index:05d}",
            text="",
            input_status=InputStatus.INVALID_RECORD,
            input_issues=issues,
            content_hash=_hash(repr(raw)),
            source_index=index,
        )

    text, status = _coerce_text(next((raw[k] for k in _TEXT_KEYS if k in raw), None), issues)
    text = clean_text(text)
    if not text and status != InputStatus.EMPTY:
        issues.append("text_blank_after_cleaning")
        status = InputStatus.EMPTY
    if len(text) > MAX_TEXT_CHARS:
        issues.append(f"text_truncated_from_{len(text)}_chars")
        text = text[:MAX_TEXT_CHARS]
        status = InputStatus.TRUNCATED

    raw_id = raw.get("id")
    msg_id = clean_text(str(raw_id))[:128] if raw_id not in (None, "") else ""
    if not msg_id:
        issues.append("id_missing")
        msg_id = f"NOID-{index:05d}-{_hash(text)[:8]}"

    brand = _clean_label(raw.get("brand"))
    if brand is None:
        issues.append("brand_missing")
    elif brand not in KNOWN_BRANDS:
        issues.append("brand_unknown")

    channel = _clean_label(raw.get("channel"))
    if channel is None:
        issues.append("channel_missing")
    elif channel not in KNOWN_CHANNELS:
        issues.append("channel_unknown")

    attachments = raw.get("attachments")
    attachment_count = len(attachments) if isinstance(attachments, (list, tuple)) else 0

    return NormalizedMessage(
        id=msg_id,
        brand=brand,
        channel=channel,
        received_at=_parse_timestamp(raw.get("received_at"), issues),
        text=text,
        attachment_count=attachment_count,
        input_status=status,
        input_issues=issues,
        content_hash=_hash(f"{brand}|{text}"),
        source_index=index,
    )


def _parse_payload(data: str) -> Tuple[List[Any], List[str]]:
    issues: List[str] = []
    data = data.lstrip("﻿")
    if not data.strip():
        return [], ["file_empty"]
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        # Fall back to JSON Lines, salvaging every line that parses.
        records: List[Any] = []
        bad = 0
        for line in data.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
        issues.append(f"file_not_valid_json ({exc.msg} at line {exc.lineno})")
        if bad:
            issues.append(f"lines_unparseable={bad}")
        return records, issues

    if isinstance(parsed, list):
        return parsed, issues
    if isinstance(parsed, dict):
        for key in ("messages", "data", "items", "records"):
            if isinstance(parsed.get(key), list):
                issues.append(f"unwrapped_top_level_key_{key}")
                return parsed[key], issues
        issues.append("top_level_object_treated_as_single_message")
        return [parsed], issues
    issues.append(f"top_level_{type(parsed).__name__}_treated_as_single_message")
    return [parsed], issues


def dedupe(messages: Iterable[NormalizedMessage]) -> Tuple[List[NormalizedMessage], List[str]]:
    """Exact duplicates (same id + same content) are dropped; id collisions with
    different content are kept but renamed so results stay addressable."""
    seen: dict[str, str] = {}
    out: List[NormalizedMessage] = []
    issues: List[str] = []
    for m in messages:
        if m.id in seen:
            if seen[m.id] == m.content_hash:
                issues.append(f"dropped_exact_duplicate:{m.id}")
                continue
            n = 2
            while f"{m.id}#{n}" in seen:
                n += 1
            m.input_issues.append(f"id_collision_renamed_from:{m.id}")
            m.id = f"{m.id}#{n}"
        seen[m.id] = m.content_hash
        out.append(m)
    return out, issues


def parse_messages(data: str) -> Tuple[List[NormalizedMessage], List[str]]:
    records, issues = _parse_payload(data)
    normalized = [normalize_record(r, i) for i, r in enumerate(records)]
    normalized, dup_issues = dedupe(normalized)
    return normalized, issues + dup_issues


def load_messages(path: str | Path) -> Tuple[List[NormalizedMessage], List[str]]:
    raw = Path(path).read_bytes()
    return parse_messages(raw.decode("utf-8", errors="replace"))
