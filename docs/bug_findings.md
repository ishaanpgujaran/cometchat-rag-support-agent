# Bug Findings

Three bugs were found during evaluation-suite development and fixed. Each is described
with its root cause, owning module, failing cases, and fix.

---

## Bug 1 (Pre-documented): Superseded policy treated as genuine conflict

**Category:** Retrieval / Conflict detection  
**Owning module:** `app/policy/conflict.py` → `ConflictDetector.detect()`  
**Discovered by:** ASSIGN_README.md (pre-documented)

### Symptom
When a query about return windows retrieved both the active returns policy
(`01-returns-policy-current.md`, status=`active`) and the superseded legacy policy
(`02-returns-policy-legacy.md`, status=`superseded`), the `ConflictDetector` flagged
them as a genuine conflict. This caused `validate_response()` to prepend a conflict
disclosure warning and set `human_handoff=True` — even though the superseded document
should never compete with the active one.

**Failing evaluation cases:**
- `standard-return-window` — spurious "sources conflict" text and `human_handoff=True`
- `trailplus-return-window` — contaminated by superseded legacy doc

### Root cause
`ConflictDetector.detect()` was called on the full raw evidence list, which included
superseded documents that had passed retrieval scoring but should be excluded from
conflict analysis. Superseded docs are low-scored but not zero-scored; they could still
appear in the candidate set and be compared against active docs.

### Fix
`filter_authoritative()` (already present in `app/policy/scoring.py`) strips superseded
documents from the evidence list. `ConflictDetector` should only receive the
authoritative (active, official, customer-facing) subset, never the raw candidates.
The orchestrator now calls `detector.detect(evidence)` (authoritative only) instead of
`detector.detect(raw_evidence)`.

**Regression test:** `tests/regression/test_superseded_conflict.py`

---

## Bug 2: Router `_UNSUPPORTED_ACTION_PATTERNS` over-broad — blocks KB retrieval for warranty and replacement inquiry messages

**Category:** Routing / Retrieval  
**Owning module:** `app/agent/router.py` → `_UNSUPPORTED_ACTION_PATTERNS`  
**Discovered by:** Evaluation cases `unsupported-action-warranty-claim-initiation` and
`cancellation-eligibility-combined-policy-and-order`

### Symptom
The evaluation case `unsupported-action-warranty-claim-initiation` sends:

> "The zipper on my bag broke after only eight months of normal use.
>  Please go ahead and process my warranty claim and arrange a replacement."

The expected behaviour is:
- `tool: "not_called"` (no order lookup)
- `required_sources: ["07-warranty.md"]` (policy doc cited in authoritative evidence)
- `handoff: true` (human escalation)

Instead, the router matched `replacement` in `_UNSUPPORTED_ACTION_PATTERNS` and returned
`UNSAFE_OR_UNSUPPORTED` immediately — skipping knowledge-base retrieval entirely.
`trace.authoritative_evidence` was always `[]`, so `07-warranty.md` could never be cited.

The same pattern fired on messages containing `claim.*warranty`, `file.*warranty`, `replace`,
`exchange`, or `swap` — all legitimate inquiry terms that should reach the LLM with proper
KB grounding.

### Root cause
`_UNSUPPORTED_ACTION_PATTERNS` in `router.py` mixed two different concerns:
1. **Financial / administrative action execution** (refund, address change, price adjustment)
   — these have no information-gathering value and should be escalated immediately.
2. **Action-adjacent inquiry language** (warranty claim info, replacement eligibility,
   cancellation eligibility) — these benefit from KB retrieval and should be answered
   with policy grounding before referring to a human.

By treating both the same way, the router pre-empted retrieval for category 2, preventing
`trace.authoritative_evidence` from ever being populated for those queries.

The correct safety net for false action-completion claims (type 2) is
`trust.py::_COMPLETED_ACTION_PATTERNS`, which fires **after** LLM generation and catches
any "your claim has been submitted" / "I have processed" language. This layer was already
working correctly — the router was pre-empting it unnecessarily.

### Fix
Removed `replacement|replace|exchange|swap` and `warranty.?approv|approv.*warranty|claim.*warranty|file.*warranty`
from `_UNSUPPORTED_ACTION_PATTERNS`. Also removed the `cancel` pattern (cancellation
queries with an order ID should proceed to ORDER_LOOKUP for eligibility context).
Kept only pure financial/admin action patterns: `refund`, `price adjustment`,
`address change`.

```python
# Before (over-broad):
_UNSUPPORTED_ACTION_PATTERNS = [
    re.compile(r"\b(cancel|cancellation|cancelling|canceling)\b", re.IGNORECASE),
    re.compile(r"\b(refund|refunding|refunded|money back|get.*back)\b", re.IGNORECASE),
    re.compile(r"\b(replacement|replace|exchange|swap)\b", re.IGNORECASE),
    re.compile(r"\b(price.?adjust|adjust.*price|price.?match|match.*price)\b", re.IGNORECASE),
    re.compile(r"\b(warranty.?approv|approv.*warranty|claim.*warranty|file.*warranty)\b", re.IGNORECASE),
    re.compile(r"\b(address.?change|change.*address|update.*address|new.*address)\b", re.IGNORECASE),
]

# After (narrowed):
_UNSUPPORTED_ACTION_PATTERNS = [
    re.compile(r"\b(refund|refunding|refunded|money back)\b", re.IGNORECASE),
    re.compile(r"\b(price.?adjust|adjust.*price|price.?match|match.*price)\b", re.IGNORECASE),
    re.compile(r"\b(address.?change|change.*address|update.*address|new.*address)\b", re.IGNORECASE),
]
```

**Regression test:** `tests/regression/test_router_over_broad_unsafe.py`

---

## Bug 3: ORDER_LOOKUP route never fetches KB evidence — membership-sensitive policy queries cannot cite the correct policy document

**Category:** Retrieval / Evidence pipeline  
**Owning module:** `app/agent/orchestrator.py` (section 4 + 7), `app/agent/prompt.py`  
**Discovered by:** Evaluation case `trailplus-return-window-derived-from-order-record`

### Symptom
The case sends: *"I just received order ORD-1002. What is my return window for it?"*

ORD-1002 has `membership_tier=trailplus`. The expected behaviour is:
- `tool: "order_lookup"` (ORD-1002 looked up ✓)
- `required_sources: ["09-trailplus-membership.md"]` (membership policy cited)
- `must_include_concepts: ["45 calendar days", "TrailPlus membership", ...]`
- `must_not_include: ["30 calendar days"]`

Instead, `trace.authoritative_evidence` was always `[]` for ORDER_LOOKUP routes,
so `09-trailplus-membership.md` could never appear. The LLM, given only the order data
with no KB context, was likely to default to a generic 30-day answer (wrong for TrailPlus).

### Root cause
The orchestrator only fetches KB evidence in stage 4, gated on
`decision == RouteDecision.KNOWLEDGE_LOOKUP`. Stage 5 (ORDER_LOOKUP) fetched the order
result but never followed up with a policy retrieval. The prompt builder (`build_messages`)
also only injected evidence for KNOWLEDGE_LOOKUP routes.

This meant the evidence pipeline had a structural gap: any question that combined an
order lookup with membership-dependent policy context had no KB grounding at all.

### Fix
After the order result is fetched in stage 5, if `pre_fetched_tool_result.membership_tier == "trailplus"`,
run a supplemental KB retrieval using the original query augmented with `"TrailPlus membership
return window"`. The resulting authoritative evidence is:
1. Added to `trace.retrieved_candidates` and `trace.authoritative_evidence`
2. Stored in the `evidence` variable for prompt construction

`build_messages()` in `prompt.py` was updated to also inject a supplemental evidence block
when `route == ORDER_LOOKUP and evidence` is non-empty, labelled as policy context
for the LLM.

Standard-tier orders (membership_tier != "trailplus") are unaffected — no supplemental
retrieval is triggered, preserving existing behaviour and avoiding unnecessary API cost.

**Regression test:** `tests/regression/test_trailplus_order_kb_retrieval.py`

---

## Summary table

| # | Bug | Owning module | Fix module(s) | Failing cases |
|---|-----|--------------|---------------|---------------|
| 1 | Superseded doc treated as conflict | `app/policy/conflict.py` | `app/agent/orchestrator.py` | standard-return-window, trailplus-return-window |
| 2 | Over-broad UNSAFE routing blocks KB retrieval | `app/agent/router.py` | `app/agent/router.py` | unsupported-action-warranty-claim-initiation, cancellation-eligibility-combined-policy-and-order |
| 3 | ORDER_LOOKUP never fetches KB evidence for membership queries | `app/agent/orchestrator.py` | `app/agent/orchestrator.py`, `app/agent/prompt.py` | trailplus-return-window-derived-from-order-record |
