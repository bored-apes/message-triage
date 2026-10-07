# Decision log

**What I noticed in the data**
- Messages hold **several requests** for **different teams** (006, 008, 014). So `intents` is a list, and each intent gets its own queue.
- **Not all senders are customers.** There is an injection asking for an INR 12,400 refund (005), invoice fraud with "bank details in the PDF" but no PDF (021), a job applicant (024) and a B2B lead (013).
- **Three can't be acted on alone:** a bare phone number (007), "what about the other one" (012) and `null` text (025, likely media).
- **Some look easy but aren't.** "Unsubscribe" (017) to a subscription brand could mean stop emails *or* stop charging me. MSG-022 cancels without saying so. The emergency (016) came by **email**.
- **Risk shows up in other forms:** a drug-interaction question (019), a third contact plus a chargeback threat (011), a wheelchair user (009), and "attached" with no attachment (021, 024).
- **Context matters:** Hinglish ("4 baje" means 16:00), relative dates that need `received_at`, and IST timestamps even for a traveller in Toronto.

**Key decisions**
- **The model proposes, code decides.** Claude returns labels, entities, flags and a draft in a schema-enforced format, with no tools. A deterministic policy table picks the queue, priority and decision. A successful injection can at worst mislabel a message. It can't choose a queue or trigger an action, and a regex layer catches injections independently.
- **Confidence is checked, not trusted.** The model's self-reported score is discounted when an extracted ID isn't in the text verbatim (invented), when regex finds an ID it missed, or when regex sees an injection it didn't flag. It is then capped for short, truncated, unknown-brand or context-free input.
- **When a human sees it.** Automation acts **only if** every intent is on a small allow-list (order status, FAQs, thank-yous, marketing opt-out), no risk flag fires, and confidence is ≥ 0.85. Otherwise:
  - emergencies → **urgent escalation**, whatever the confidence;
  - confidence ≥ 0.60 → **route to the owning team**;
  - lower → **human review** of the label.
- **Degrade, don't die.** On an API failure, refusal or bad output, a keyword fallback routes everything to humans; it is capped so it can never auto-resolve. Auth errors trip a circuit breaker.

**What I chose not to build**
- **Integrations** (order/booking lookup, sending replies, CRM): the output is a decision plus a draft. FAQ auto-replies need a knowledge base before they can quote real hours or prices.
- **Thread history / customer identity.** This is the biggest gap and needs channel APIs.
- **Attachments and voice notes** (OCR, transcription).
- **Production plumbing:** durable queue, endpoint auth, PII redaction before text reaches the API (needs a DPA / DPDP review).
- **Calibration:** 25 messages is not a labelled set.

**Where this breaks**
- **Thresholds are judgement, not calibration.** Model confidence clusters high, so the grounding checks do the honest work.
- **Regex flags are English-centric.** A Hindi threat or injection relies on the model alone. False positives ("medication") over-route to humans: that costs headcount, not safety.
- **Date arithmetic is the model's and is unverified.** Context-dependent messages always need a human.
- **A long outage at 10k/day floods humans.** The routing is right; the staffing isn't.

**With another day**
- **Calibrate:** label ~300 real messages, calibrate per-intent thresholds, and run shadow mode next to agents for two weeks before enabling auto-resolve.
- **Context:** add thread lookup and KB-grounded FAQ answers.
- **Cost:** measured at $17.46/1k on a cold 25-message run (about $14.6/1k steady state). Run an Opus vs Sonnet eval on the labels, and use the Batch API (−50%) for non-urgent traffic.
- **Operations:** queue plus idempotency, and alerts on fallback rate and human-review rate.

**AI tools used**
- **claude-opus-5-5 (API):** the classifier inside the service.
