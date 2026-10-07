# Message triage service

This service reads inbound customer messages (WhatsApp, email, Instagram) for several brands. For each message it decides:

- **what the customer wants:** one or more intents;
- **what useful information it contains:** order ids, booking refs, phones, amounts, dates and similar;
- **what happens next and who handles it:** a decision, a queue, a priority, an SLA (target response time) and a draft reply.

Every result carries a confidence score and the reasons behind the decision.

Start with [DECISION_LOG.md](DECISION_LOG.md).

## Quick start

You need Python 3.10 or newer, because the `anthropic` 1.x SDK requires it.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # the only thing you need to supply
python -m triage data/messages.json        # writes out/results.jsonl + out/summary.json
```

You can run it without an API key:

```bash
python -m triage data/messages.json --offline
```

Offline mode is the degraded mode the service also falls back to when the API is down. It uses deterministic rules only, and every message goes to a human.

Other commands:

```bash
python -m triage.evaluate out/results.jsonl   # score against hand labels in data/expected.json
python -m triage.cost                         # cost estimate per 1,000 messages
python -m triage.cost out/results.jsonl       # measured cost from a real run
pytest                                        # 55 behavioural tests, no API calls
```

### HTTP service

```bash
uvicorn triage.api:app --port 8000
curl -s localhost:8000/triage -d '{"id":"X1","brand":"hair-studio","channel":"instagram","text":"are you open on sundays"}'
curl -s localhost:8000/triage/batch --data-binary @data/messages.json
```

The endpoints never return a 500 for bad input. A broken body becomes a result that is routed to a human.

### Docker

```bash
docker build -t triage .
docker run --rm -e ANTHROPIC_API_KEY -p 8000:8000 triage
docker run --rm -e ANTHROPIC_API_KEY -v "$PWD/out:/app/out" triage python -m triage data/messages.json
```

### Configuration (environment variables)

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Required for model classification. Without it, the service runs in rules mode. |
| `TRIAGE_MODEL` | `claude-opus-5-5` | `claude-sonnet-5-5` and `claude-haiku-4-5` are also supported. See the cost section below. |
| `TRIAGE_EFFORT` | `low` | Classification doesn't benefit from deep reasoning. |
| `TRIAGE_CONCURRENCY` | `8` | Number of parallel API calls. |
| `TRIAGE_REFUSAL_FALLBACK` | `1` | Server-side fallback if the model declines a request. Set to `0` on Bedrock, Vertex or Foundry. |

Thresholds, queues and per-intent policy live in [triage/config.py](triage/config.py).

## How it works

```
raw record ─► ingest (normalise, never raise) ─► signals (regex: ids, injection, chargeback, health, stranded …)
                                                       │
                                     Claude (structured JSON, no tools) ──fail──► rules fallback
                                                       │
                     reconcile (ground entities, add missed ids, union flags) ─► confidence ─► routing policy
```

| File | Responsibility |
|---|---|
| [triage/ingest.py](triage/ingest.py) | Accepts JSON, JSON Lines, a wrapped object or garbage. Handles `null`, number, list and object text, missing or duplicate ids, invisible characters, control characters and very long bodies. Every issue is recorded in `input_issues`. |
| [triage/signals.py](triage/signals.py) | Deterministic extraction and risk flags. These don't depend on a model that the message may be trying to manipulate. |
| [triage/llm.py](triage/llm.py) | Calls Claude with the prompt and a JSON schema. The customer text is delimited and escaped. The model has no tools, and the prompt prefix is cached. |
| [triage/policy.py](triage/policy.py) | Confidence fusion and **the routing rule**. |
| [triage/rules.py](triage/rules.py) | Keyword classifier used offline or when the API fails. Its confidence is capped, so it can never auto-resolve. |
| [triage/pipeline.py](triage/pipeline.py) | Concurrency, one retry on bad output, a circuit breaker on auth errors, and per-message error isolation. |

### Output shape (one JSON line per message)

```jsonc
{
  "message_id": "MSG-008", "brand": "vitalis-wellness", "channel": "email",
  "input_status": "ok", "input_issues": [],
  "analysis": {
    "language": "en",
    "intents": [
      {"intent": "refund_request", "summary": "…", "next_step": "…", "confidence": 0.95},
      {"intent": "new_purchase",   "summary": "…", "next_step": "…", "confidence": 0.9}
    ],
    "entities": [{"type": "order_id", "value": "VW-48190", "normalized": "VW-48190", "grounded": true, "source": "model"}],
    "urgency": "normal", "sentiment": "negative",
    "risk_flags": ["money_movement"], "missing_info": [], "draft_reply": "…", "rationale": "…"
  },
  "routing": {
    "decision": "route_to_team",           // auto_resolve | route_to_team | human_review | urgent_escalation
    "primary_queue": "vitalis-wellness.support",
    "priority": "normal", "sla_minutes": 480,
    "tasks": [ {"intent": "refund_request", "queue": "vitalis-wellness.support", "automatable": false, …},
               {"intent": "new_purchase",   "queue": "vitalis-wellness.sales",   "automatable": false, …} ],
    "reasons": ["human-required flags: money_movement", "intents needing a human: new_purchase, refund_request"],
    "send_draft_reply": false
  },
  "confidence": {"score": 0.9, "band": "high",
                 "components": {"model_self_reported": 0.9, "grounding_factor": 1.0, "recall_factor": 1.0, "safety_agreement_factor": 1.0},
                 "caps_applied": []},
  "meta": {"classifier": "llm", "model": "claude-opus-5-5", "prompt_version": "triage-v1.3",
           "latency_ms": 4100, "input_tokens": 260, "output_tokens": 690, "cache_read_tokens": 2290, "cost_usd": 0.0153, "errors": []}
}
```

Why this shape:

- **Intents are a list.** Multi-request messages (MSG-006, 008, 014) fan out into one task per intent, each with its own queue.
- **Analysis is separate from routing.** What the model said is kept apart from what the service decided.
- **Every decision carries `reasons`.** An agent can see why a message reached them.

## Confidence and the human-in-the-loop rule

`confidence.score` combines the model's self-reported confidence with independent checks:

- **Grounding:** every extracted id must appear verbatim in the text. Each invented id costs ×0.75.
- **Recall:** each id that the regex found but the model missed costs ×0.9.
- **Safety agreement:** if the regex detected an injection that the model didn't flag, the score is halved.
- **Input caps:**
  - rules fallback: at most 0.55
  - 3 words or fewer: at most 0.6
  - unknown brand: at most 0.7
  - truncated or non-string text: at most 0.75
  - message needs context: at most 0.5

**Automation acts (`auto_resolve`) only if all of these hold:**

1. the text is readable and the brand is known;
2. every intent is on the automatable allow-list: `order_status`, `product_question`, `pricing_question`, `business_info`, `membership_question`, `marketing_unsubscribe`, `positive_feedback`;
3. no human-required risk flag is present, whether it came from the model or the regex layer;
4. confidence is at least 0.85, and the model (not the rules fallback) produced the classification.

**Otherwise a human sees it:**

| Decision | When |
|---|---|
| `urgent_escalation` | An emergency is in progress (`time_critical` / `travel_disruption`). This applies whatever the confidence. |
| `route_to_team` | Confidence is at least 0.60. The label is trusted, and the owning team does the work. |
| `human_review` | Confidence is below 0.60, or the input is unreadable. A triage agent checks the label first. |

Flags that always need a human: `prompt_injection`, `payment_fraud_risk`, `money_movement`, `health_safety`, `chargeback_threat`, `legal_threat`, `repeat_contact`, `vulnerable_customer`, `time_critical`, `missing_attachment`, `missing_context`, `ambiguous_request`.

## Scale and cost: 10,000 messages a day

**Throughput.** 10,000 messages a day is about 7 a minute on average; plan for a 10× peak of about 1.2 a second. At roughly 4–6 s per call, that is about 6–8 calls in flight, which the default concurrency covers. One small container handles it, so the API bill dominates the cost. For higher volume, run several workers behind a queue. The pipeline is stateless per message.

**Measured run** (all 25 messages, `claude-opus-5-5`, effort `low`, 24 billed calls; `samples/live/summary.json`):

- **$0.419 in total, which works out to $17.46 per 1,000 messages.**
- That figure includes 8 parallel cold-start cache writes: 31,256 tokens, costing $0.16. A 25-message run pays these up front; at 10k/day the cache stays warm.
- **Per call:** about 3,907 cached prefix tokens, 156 uncached input tokens and 490 output tokens (thinking included).

**Steady-state cost per 1,000 messages** (`python -m triage.cost`, using the measured token counts plus headroom):

| Model | $/1k msgs | $/day at 10k | $/month | with Batch API (−50%) |
|---|---|---|---|---|
| **claude-opus-5-5** (default) | **$14.59** | $146 | ~$4,380 | $7.30 |
| claude-sonnet-5-5 | $7.68 | $77 | ~$2,300 | $3.84 |
| claude-haiku-4-5 | $6.78 | $68 | ~$2,040 | $3.39 |

How the number is built, per message:

- **Input:** 3,900 prefix tokens (system prompt plus JSON-schema overhead), read from cache at $0.20 per million tokens on Opus 5.5. 250 uncached tokens, against 156 measured; the budget allows for longer real emails.
- **Output:** 600 tokens, against 490 measured. Output is about 80% of the cost.
- **Retries:** 2% re-asked after invalid output.
- **Cache writes:** at worst one cache write every 5 minutes (about 7 messages a minute keeps the cache warm).
- **Haiku 4.5:** the 3.9k prefix is shorter than its 4,096-token minimum for caching, so the full prompt is billed on every call. That is why Haiku saves little over Sonnet here.

Assumptions to check:

- Prices are Anthropic API list prices as of 2026-10.
- Infrastructure cost (about $30–50 a month for one small container) is ignored.

**Levers, in order:**

1. **Batch API for non-urgent traffic (−50%).** The regex layer already fast-paths emergencies, so batching the rest is safe.
2. **Drop `draft_reply` for intents that never auto-send.** This saves about 30% of output tokens.
3. **Move to Sonnet 5.5 if an eval on labelled data shows equal routing quality.**

I kept Opus as the default because a misroute on money, health or fraud messages costs more than the difference.

## Running against the provided file

- **`samples/offline/`** is a committed rules-mode run. It produces 0 unsafe auto-resolves and 0 missed emergencies, and sends everything else to humans by design.
- **`samples/live/`** is the committed Claude Opus 5.5 run. Re-produce it with:

  ```bash
  python -m triage data/messages.json --out samples/live
  ```

  Then score it with:

  ```bash
  python -m triage.evaluate samples/live/results.jsonl
  ```
