# Aster & Row — Bug Diary & Evaluation Evolution

This diary documents the failure modes discovered during the development and evaluation of the
Aster & Row RAG Customer Support Agent, their root causes, architectural fixes, and regression tests.

---

## Evaluation Summary

- **Baseline Result (Early Run):** 13 / 25 passed (52% pass rate; 12 failing cases)
- **Final Result:** **25 / 25 passed (100% pass rate; 0 failing cases)**
- **Unit & Regression Suite:** 224 / 224 passed (`pytest tests/`)

---

## Documented Bugs & Root Cause Analyses

### Bug 1: Superseded Policy Document Treated as Active Conflict
**Category:** Retrieval / Document Precedence
**Reproduction:** Querying return windows retrieved both `01-returns-policy-current.md` (active, 30 days) and `02-returns-policy-legacy.md` (superseded, 60 days).
**Actual failure:** The conflict detector flagged them as an active conflict, causing `validate_response()` to prepend a conflict warning and trigger `human_handoff=True`.
**Root cause:** Raw retrieved candidates were passed to `ConflictDetector.detect()` in `app/agent/orchestrator.py` before filtering for authoritative policy documents.
**Fix:** Filtered candidates with `filter_authoritative()` in `app/policy/scoring.py` prior to conflict detection. Superseded documents carry `status=superseded` and are penalized in precedence scoring rather than triggering human escalation.
**Regression test:** `tests/regression/test_superseded_conflict.py`

---

### Bug 2: False-Positive Conflict Detections Across Distinct Document Contexts *(Discovered beyond visible case wording)*
**Category:** Conflicting Policy Answers & Source Independence
**Reproduction:** Queries regarding standard return window (`01-returns-policy-current.md`), TrailPlus window (`09-trailplus-membership.md`), or Canadian shipping rates (`06-shipping-international.md`).
**Actual failure:** The agent prepended `⚠️ I found conflicting information between...` and set `human_handoff=True` on standard return window queries, TrailPlus queries, and Canada multi-turn queries.
**Root cause:** A heuristic "Tier 2 numeric fallback" in `app/policy/conflict.py` compared numbers across any two active documents sharing broad keywords (`"return window"`, `"trailplus"`, `"warranty"`). It treated legitimate policy distinctions (e.g., standard 30-day window in `01` vs. TrailPlus 45-day window in `09`, or domestic shipping `05` vs. international shipping `06`) as conflicting claims.
**Fix:** Removed the brittle numeric fallback heuristic in `app/policy/conflict.py` and designated `CONFLICT_REGISTRY` (Tier 1) as the sole authoritative source of genuine active conflicts (specifically `11-product-care.md` vs. `12-breeze-tumbler-product-card.md`).
**Regression test:** `tests/regression/test_superseded_conflict.py`

---

### Bug 3: Gemini SDK Function Calling `thought_signature` Stripping
**Category:** Tool Reliability & API Integration
**Reproduction:** Calling tool lookup functions on multi-turn queries or edge-case lookups (e.g. `unknown-order`, `order-id-case-and-whitespace-normalization`).
**Actual failure:** API crashed immediately with `ClientError: 400 INVALID_ARGUMENT. Function call is missing a thought_signature in functionCall parts.`
**Root cause:** In `app/agent/orchestrator.py` (`_call_gemini_with_tools`), when the model emitted a tool call, a new `genai_types.Content` object was reconstructed from scratch, stripping the internal `thought_signature` attribute required by Gemini 2.5/3.x reasoning models on the second turn.
**Fix:** Reused `candidate.content` directly when appending the model's function call turn into the follow-up message payload in `app/agent/orchestrator.py`.
**Regression test:** `tests/integration/test_orchestrator.py`

---

### Bug 4: False Completed-Action Redaction for Valid Order Statuses *(Discovered beyond visible case wording)*
**Category:** Safety / Trust Layer & Tool Reliability
**Reproduction:** Inquiring about order `ORD-1004` (cancelled order in `data/orders.json`).
**Actual failure:** The agent accurately retrieved `status: cancelled`. However, when stating *"Your order was cancelled and will not be shipped"*, the safety validator matched `order is cancelled`, replaced the answer with refusal text, and forced an unwanted human handoff.
**Root cause:** Overly broad regex in `app/safety/trust.py` (`_COMPLETED_ACTION_PATTERNS`) conflated passive factual descriptions of database records (`"order was cancelled"`, `"ticket can be opened"`) with first-person unverified action completion claims (`"I have cancelled your order"`, `"I processed your refund"`).
**Fix:** Refined regex patterns in `app/safety/trust.py` to strictly target first-person active claims (`"I have cancelled"`, `"We processed your refund"`).
**Regression test:** `tests/integration/test_orchestrator.py`

---

### Bug 5: Over-broad `NEEDS_ORDER_ID` Router Signal Hijacking Policy Inquiries
**Category:** Router & Multi-Turn Context Management
**Reproduction:** Conversational inquiries containing words like "order" or "bought" without an ID (e.g., *"I bought something recently and I changed my mind. How long do I have to send it back?"*).
**Actual failure:** Router misrouted the request to `NEEDS_ORDER_ID`, refusing to answer policy questions and demanding an order ID.
**Root cause:** Router Signal in `app/agent/router.py` matched individual words like `"bought"`, `"order"`, `"purchase"` without checking whether the user was asking for specific order tracking or a general policy question.
**Fix:** Replaced single-word matching in `app/agent/router.py` with intent regex patterns (`_ORDER_STATUS_INTENT_PATTERNS`) targeting tracking and order lookup requests, allowing policy inquiries with conversational words to reach `KNOWLEDGE_LOOKUP`.
**Regression test:** `tests/regression/test_router_over_broad_unsafe.py`

---

### Bug 6: ORDER_LOOKUP Omitted Supplemental KB Retrieval for Policy Queries
**Category:** Multi-Source Grounding (Orders + Knowledge Base)
**Reproduction:** Customer asking a return-window or cancellation policy question that referenced a specific order (e.g., `ORD-1002` with TrailPlus tier).
**Actual failure:** `trace.authoritative_evidence` was empty and policy details were ungrounded because KB retrieval was skipped entirely for `ORDER_LOOKUP` routes.
**Root cause:** `app/agent/orchestrator.py` treated `ORDER_LOOKUP` and `KNOWLEDGE_LOOKUP` as mutually exclusive, never querying the knowledge base during order lookups.
**Fix:** Added supplemental KB retrieval on `ORDER_LOOKUP` routes in `app/agent/orchestrator.py` whenever the order record carries `membership_tier == "trailplus"` or when the query asks about order-level policies (cancellation, returns, warranty), providing both the order result and authoritative policy chunks to the model.
**Regression test:** `tests/regression/test_trailplus_order_kb_retrieval.py`

---

### Bug 7: Stale Historical Delivery Timestamps Leaked on Terminal Returned Orders
**Category:** Tool Reliability / Order Projection
**Reproduction:** Inquiring about order `ORD-1008` (*"Can you tell me what is happening with order ORD-1008?"* which has `status: "returned"` in `data/orders.json`).
**Actual failure:** The agent outputted `"Your order was delivered on July 25, and the return was received and processed"`, leaking stale delivery dates on a returned order.
**Root cause:** In `data/orders.json`, `ORD-1008` is in a terminal `returned` status but retains historical shipping and delivery timestamps. `_apply_status_rules()` in `app/orders/lookup.py` nullified `carrier`, `tracking_number`, and `estimated_delivery` for terminal statuses, but forgot to nullify `shipped_at` and `delivered_at`.
**Fix:** Updated `_apply_status_rules()` in `app/orders/lookup.py` to suppress both `shipped_at = None` and `delivered_at = None` for any terminal order status (`cancelled`, `returned`).
**Regression test:** `tests/unit/test_orders.py`

---

### Bug 8: "internal" Word False Positive in Forbidden-Field Validator *(Discovered beyond visible case wording)*
**Category:** Safety / Trust Layer & Privacy Enforcement
**Reproduction:** Generating legitimate descriptive English prose explaining policy boundaries (e.g., *"The migration note is an internal document and not authoritative policy"*).
**Actual failure:** The forbidden-field validator detected the bare word `"internal"`, flagged a false-positive data leak, and redacted the entire sentence.
**Root cause:** `_FORBIDDEN_FIELDS` in `app/safety/trust.py` matched the bare word `\binternal\b` anywhere in the generated output text, failing to distinguish between descriptive English adjectives and actual database field disclosures.
**Fix:** Narrowed the check for `"internal"` in `app/safety/trust.py` to data-disclosure contexts (`\binternal\s+(?:note|notes|field|fields|data|record|score|tag|tags|flag|flags)\b`) and defined `ALLOWED_INTERNAL_PHRASES` (`"internal document"`, `"internal content"`, `"internal policy"`, `"internal migration"`, `"internal material"`), ensuring descriptive prose is permitted while actual internal field leaks remain strictly redacted.
**Regression test:** `tests/regression/test_internal_word_false_positive.py`

---

### Bug 9: Router Context Bleed — Order Session ID Reused for Unrelated Queries *(Discovered beyond visible case wording)*
**Category:** Router & Multi-Turn Context Management
**Reproduction:** In a session where `ORD-1007` was just looked up, ask *"Do you ship internationally?"* — a pure knowledge question.
**Actual failure:** Router reused `last_order_id` from session context, routed to `ORDER_LOOKUP`, and looked up `ORD-1007` again. The answer was about international shipping but was grounded incorrectly via order context.
**Root cause:** Session context reuse logic in `app/agent/router.py` fired whenever `last_order_id` existed, without checking whether the new message contained any order-referencing signal.
**Fix:** Added `ORDER_CONTINUATION_SIGNALS` guard in `app/agent/router.py` — `last_order_id` is only reused when message contains order-referencing vocabulary or pronouns.
**Regression test:** `tests/regression/test_router_no_context_bleed.py`

---

### Bug 10: LLM Hallucinating Citation Headings for Order Tool Responses *(Discovered beyond visible case wording)*
**Category:** Citation & Prompt Alignment
**Reproduction:** Ask *"Where is order ORD-1007?"* — the system prompt instructs the model to always cite sources, but there is no KB document to cite.
**Actual failure:** Model generated fake citations like `"01-returns-policy-current.md#Order Status Lookup"` (heading does not exist) or `"tool_result#order_lookup"`. Validator stripped them, leaving customer with zero attribution or broken bracket artifacts.
**Root cause:** The system prompt instructed the model to cite sources without (a) specifying which citations are valid, or (b) handling the case where the answer source is an order tool result rather than a KB document.
**Fix:** Orchestrator in `app/agent/orchestrator.py` now adds `Order record: {order_id}` as a deterministic citation for `ORDER_LOOKUP` responses. Prompt in `app/agent/prompt.py` now passes an explicit anchor list of valid citation strings and explicitly instructs the model not to cite the order tool result as a KB document.
**Regression test:** `tests/regression/test_citation_artifact_cleanup.py`
