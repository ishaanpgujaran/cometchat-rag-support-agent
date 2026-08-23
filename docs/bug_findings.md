# Bug Findings & Evaluation Diary

This diary documents the failure modes discovered during the development and evaluation of the
Aster & Row RAG Support Agent, their root causes, the changes made, and the regression tests
that verify each fix.

---

## Baseline Evaluation Summary (Early Run)

- **Total cases evaluated:** 25 (15 visible + 10 original test suite)
- **Early Baseline Result:** 13 / 25 passed (52% pass rate; 12 failing cases)
- **Primary failure symptoms observed:**
  1. Spurious conflict warnings and forced human handoffs on standard return, TrailPlus, and Canadian shipping queries.
  2. Gemini SDK `ClientError: 400 INVALID_ARGUMENT (missing thought_signature)` during multi-turn function calling.
  3. False-positive completed-action redaction stripping accurate order status ("order is cancelled") and warranty guidance.
  4. Misrouted conversational return policy queries (e.g. "I bought something...", "I placed an order last week...") to `NEEDS_ORDER_ID` instead of `KNOWLEDGE_LOOKUP`.

---

## Documented Bugs & Root Cause Analyses

### Bug 1: Superseded Policy Document Treated as Active Conflict
- **Category:** Retrieval / Conflict Resolution (ASSIGN_README Core Requirement 1)
- **Owning Module:** `app/policy/conflict.py` & `app/agent/orchestrator.py`
- **Symptom:** When a query regarding return windows retrieved both `01-returns-policy-current.md` (active) and `02-returns-policy-legacy.md` (superseded), the conflict detector flagged them as a genuine conflict, causing `validate_response()` to prepend a conflict warning and trigger `human_handoff=True`.
- **Root Cause:** Raw retrieved candidates were passed to `ConflictDetector.detect()` before filtering for authoritative policy documents.
- **Fix:** Filtered candidates with `filter_authoritative()` prior to conflict detection. Superseded documents carry `status=superseded` and are penalized in precedence scoring rather than triggering human escalation.
- **Regression Test:** `tests/regression/test_superseded_conflict.py`

---

### Bug 2: False-Positive Conflict Detections Across Distinct Document Contexts (Discovered Beyond Exact Visible Case Wording)
- **Category:** Conflicting Policy Answers & Source Independence
- **Owning Module:** `app/policy/conflict.py` (`Tier 2 numeric fallback`)
- **Symptom:** The agent prepended `⚠️ I found conflicting information between...` and set `human_handoff=True` on standard return window queries, TrailPlus queries, and Canada multi-turn queries.
- **Root Cause:** A heuristic "Tier 2 numeric fallback" compared numbers across any two active documents sharing broad keywords (`"return window"`, `"trailplus"`, `"warranty"`). It treated legitimate policy distinctions (e.g., standard 30-day window in `01` vs. TrailPlus 45-day window in `09`, or domestic shipping `05` vs. international shipping `06`) as conflicting claims.
- **Fix:** Removed the brittle numeric fallback heuristic and designated `CONFLICT_REGISTRY` (Tier 1) as the sole authoritative source of genuine active conflicts (specifically `11-product-care.md` vs. `12-breeze-tumbler-product-card.md`).
- **Regression Test:** `tests/regression/test_superseded_conflict.py` & `tests/unit/test_policy.py`

---

### Bug 3: Gemini SDK Function Calling `thought_signature` Stripping
- **Category:** Tool Reliability & API Integration
- **Owning Module:** `app/agent/orchestrator.py` (`_call_gemini_with_tools`)
- **Symptom:** `unknown-order` and `order-id-case-and-whitespace-normalization` crashed immediately with:
  `ClientError: 400 INVALID_ARGUMENT. Function call is missing a thought_signature in functionCall parts.`
- **Root Cause:** When the model emitted a tool call, `orchestrator.py` manually reconstructed a new `genai_types.Content` object from scratch. For Gemini 2.5/3.x models with internal reasoning/thinking enabled, this stripped the internal `thought_signature` attribute required by Google's API on the second turn.
- **Fix:** Reused `candidate.content` directly when appending the model's function call turn into the follow-up message payload.
- **Regression Test:** `tests/unit/test_observability.py` & `tests/integration/test_orchestrator.py`

---

### Bug 4: False Completed-Action Redaction for Valid Order Statuses (Discovered Beyond Exact Visible Case Wording)
- **Category:** Safety / Trust Layer & Tool Reliability
- **Owning Module:** `app/safety/trust.py` (`_COMPLETED_ACTION_PATTERNS`)
- **Symptom:** For order `ORD-1004` (cancelled order), the agent accurately retrieved `status: cancelled` from `data/orders.json`. However, when stating *"Your order was cancelled and will not be shipped"*, the safety validator matched `order is cancelled`, replaced the answer with refusal text, and forced an unwanted human handoff.
- **Root Cause:** The regex `_COMPLETED_ACTION_PATTERNS` was overly broad, conflating passive factual descriptions of existing database records (`"order was cancelled"`, `"ticket can be opened"`) with first-person unverified action completion claims (`"I have cancelled your order"`, `"I processed your refund"`).
- **Fix:** Refined regex patterns to strictly target first-person active claims (`"I have cancelled"`, `"We processed your refund"`, `"Replacement is on its way"`).
- **Regression Test:** `tests/unit/test_observability.py::TestPrivacySecurity` & `tests/integration/test_orchestrator.py`

---

### Bug 5: Over-broad `NEEDS_ORDER_ID` Router Signal Hijacking Policy Inquiries
- **Category:** Router & Multi-Turn Context Management
- **Owning Module:** `app/agent/router.py` (`_ORDER_CONTEXT_KEYWORDS`)
- **Symptom:** Inquiries such as *"I bought something recently and I changed my mind. How long do I have to send it back?"* and *"I placed an order last week before I had a TrailPlus membership. Does my order qualify for the 45-day window?"* were misrouted to `NEEDS_ORDER_ID` instead of retrieving policy documents.
- **Root Cause:** Router Signal 4 matched individual words like `"bought"`, `"order"`, `"purchase"` without checking whether the user was asking for specific order tracking or a general policy question.
- **Fix:** Replaced single-word matching with intent regex patterns (`_ORDER_STATUS_INTENT_PATTERNS`) targeting tracking and order lookup requests, allowing policy inquiries with conversational words to reach `KNOWLEDGE_LOOKUP`.
- **Regression Test:** `tests/regression/test_router_over_broad_unsafe.py`

---

### Bug 6: ORDER_LOOKUP Omitted Supplemental KB Retrieval for Policy Queries
- **Category:** Multi-Source Grounding (Orders + Knowledge Base)
- **Owning Module:** `app/agent/orchestrator.py` & `app/agent/prompt.py`
- **Symptom:** When a customer asked a return-window or cancellation policy question that referenced a specific order (e.g. `ORD-1002` with TrailPlus tier), `trace.authoritative_evidence` was empty because KB retrieval was skipped for `ORDER_LOOKUP` routes.
- **Fix:** Added supplemental KB retrieval on `ORDER_LOOKUP` routes whenever the order record carries `membership_tier == "trailplus"` or when the query asks about order-level policies (cancellation, returns, warranty), providing both the order result and authoritative policy chunks to the model.
- **Regression Test:** `tests/regression/test_trailplus_order_kb_retrieval.py`

---

## Final Evaluation Summary

- **Total cases evaluated:** 25 (15 visible + 10 original test suite)
- **Target Pass Rate:** 100% across all 10 evaluation categories:
  - Retrieval
  - Groundedness
  - Tool use
  - Tool arguments
  - Privacy
  - Multi-turn
  - Safety
  - Abstention
  - Citation
  - Conflict handling
