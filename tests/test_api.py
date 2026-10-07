from __future__ import annotations

from fastapi.testclient import TestClient

from triage import api
from triage.pipeline import TriagePipeline


def client() -> TestClient:
    api._pipeline = TriagePipeline(None)  # offline for tests
    return TestClient(api.app)


def test_health():
    r = client().get("/healthz")
    assert r.status_code == 200 and r.json()["mode"] == "rules"


def test_single_valid():
    r = client().post("/triage", json={"id": "X", "brand": "hair-studio", "channel": "instagram", "text": "are you open on sundays"})
    assert r.status_code == 200
    assert r.json()["message_id"] == "X"


def test_single_garbage_body_is_not_a_500():
    for body in [b"", b"{{{", b"null", b"[1,2]", "plain text message".encode()]:
        r = client().post("/triage", content=body)
        assert r.status_code == 200, body


def test_batch():
    r = client().post("/triage/batch", content=b'[{"id":"A","text":"Unsubscribe"},{"id":"B","text":null}]')
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2
