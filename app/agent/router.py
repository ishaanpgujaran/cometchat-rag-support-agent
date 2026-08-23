"""
app/agent/router.py
-------------------
Deterministic message router for the support-agent orchestration layer.

Design
~~~~~~
All routing decisions are made WITHOUT LLM calls.  The cascade below fires
in strict priority order — the first matching signal wins.

Signal cascade
~~~~~~~~~~~~~~
1. Safety / unsupported-action gate
   Regex + keyword scan for: cancel, refund, replacement, price adjustment,
   warranty approval, address change, fraud, account takeover, safety issue,
   legal demand, privacy request, exposing internals / prompt / credentials.
   → UNSAFE_OR_UNSUPPORTED (human_handoff required)

2. Order-ID present in message
   Regex for ORD-\\d{4,} → ORDER_LOOKUP

3. Multi-turn order continuation (no new ID)
   last_route == ORDER_LOOKUP AND last_order_id is set AND new message
   contains delivery/status keywords → ORDER_LOOKUP (reuse cached ID)

4. Order-context keyword with no ID and no prior order
   "order", "package", "tracking", "delivery", "shipped", "arrive" etc.
   → NEEDS_ORDER_ID

5. Multi-turn knowledge follow-up
   last_route == KNOWLEDGE_LOOKUP AND (message is short OR shares keywords
   with last_topic) → KNOWLEDGE_LOOKUP  (query is augmented with last_topic)

6. Catch-all → KNOWLEDGE_LOOKUP (let retrieval decide; ABSTAIN_NO_EVIDENCE
   is set by the orchestrator if the evidence set is empty/non-authoritative)

Public API
~~~~~~~~~~
    RouteDecision           — enum of possible routing outcomes
    RouteResult             — dataclass carrying decision + resolved query
    route(session, message) -> RouteResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.session.store import Session


# ---------------------------------------------------------------------------
# RouteDecision enum
# ---------------------------------------------------------------------------

class RouteDecision(str, Enum):
    """All possible routing outcomes."""
    KNOWLEDGE_LOOKUP      = "KNOWLEDGE_LOOKUP"
    ORDER_LOOKUP          = "ORDER_LOOKUP"
    NEEDS_ORDER_ID        = "NEEDS_ORDER_ID"
    UNSAFE_OR_UNSUPPORTED = "UNSAFE_OR_UNSUPPORTED"
    ABSTAIN_NO_EVIDENCE   = "ABSTAIN_NO_EVIDENCE"


# ---------------------------------------------------------------------------
# RouteResult
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    """Output of route().

    Attributes
    ----------
    decision : RouteDecision
        The primary routing decision.
    query : str
        The (possibly augmented) query string to use for retrieval.
        For non-knowledge routes this mirrors the original message.
    resolved_order_id : str | None
        Populated when decision == ORDER_LOOKUP.  Already normalised
        (uppercase, stripped).
    human_handoff : bool
        True when the decision itself mandates escalation (UNSAFE path).
    handoff_reason : str | None
        Short human-readable reason for the handoff (if applicable).
    """
    decision: RouteDecision
    query: str
    resolved_order_id: Optional[str] = None
    human_handoff: bool = False
    handoff_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Canonical order-ID format: ORD- followed by 4+ digits (case-insensitive)
_ORDER_ID_RE = re.compile(r"\bORD-\d{4,}\b", re.IGNORECASE)

# Delivery/status follow-up keywords (multi-turn order continuation)
_ORDER_FOLLOWUP_KEYWORDS: frozenset[str] = frozenset({
    "arrive", "arrival", "delivery", "deliver", "delivered",
    "shipped", "shipping", "ship", "status", "tracking", "track",
    "package", "parcel", "when", "where", "eta", "estimated",
    "update", "late", "delay", "delayed", "receive", "received",
})

# Patterns that indicate the customer specifically wants to check/track their order
# but hasn't provided an order ID yet.
_ORDER_STATUS_INTENT_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(where\s+is|status\s+of|track|tracking\s+number|tracking\s+info|when\s+will.*arrive|has.*shipped|check.*status|update\s+on)\b.{0,30}\b(my|the|an)?\s*(order|package|parcel|shipment|delivery)\b", re.IGNORECASE),
    re.compile(r"\b(my|the)\s+(order|package|parcel|shipment)\s+(status|tracking|hasn't\s+arrived|is\s+late|delayed)\b", re.IGNORECASE),
    re.compile(r"\b(track|lookup|look\s+up|find)\s+(my|the)?\s*(order|package|shipment)\b", re.IGNORECASE),
]

_DAMAGED_ITEM_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(damaged|broken|defect|defective|arrived\s+broken|broken\s+zipper|tear|torn)\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Unsupported / unsafe action patterns
# Encoded from Doc 13 (13-support-escalation.md) internal rules.
# ---------------------------------------------------------------------------

# Actions the system cannot execute → always escalate.
#
# NARROWING RATIONALE (Bug 2 fix):
# The original patterns were over-broad: `replacement|replace|exchange|swap`
# fired on messages like "arrange a replacement" (warranty context) BEFORE
# retrieval could surface the relevant policy doc.  Similarly,
# `claim.*warranty|file.*warranty` fired on legitimate warranty-info queries.
#
# The correct safety net for false action-completion claims is
# trust.py::_COMPLETED_ACTION_PATTERNS — which fires AFTER the LLM responds.
# Routing to UNSAFE pre-LLM is only appropriate for pure financial/admin action
# demands (refund, address change, price adjustment) where NO policy context
# is useful, or for explicit warranty-approval execution phrasing.
#
# warranty/replacement INQUIRY (e.g. "is my zipper covered?", "can I get a
# replacement?") → KNOWLEDGE_LOOKUP so that the policy doc can be cited.
# warranty/replacement ACTION EXECUTION (e.g. "process my claim right now") →
# handled by trust.py::_COMPLETED_ACTION_PATTERNS after retrieval.
_UNSUPPORTED_ACTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(cancel|cancellation|cancelling|canceling)\b", re.IGNORECASE),
    re.compile(r"\b(refund|refunding|refunded|money back)\b", re.IGNORECASE),
    re.compile(r"\b(price.?adjust|adjust.*price|price.?match|match.*price)\b", re.IGNORECASE),
    re.compile(r"\b(address.?change|change.*address|update.*address|new.*address)\b", re.IGNORECASE),
]

# Safety / legal / privacy / adversarial probing → always escalate
_SAFETY_ESCALATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(fraud|fraudulent|scam|scammed|stolen|unauthorized.charge)\b", re.IGNORECASE),
    re.compile(r"\b(account.?takeover|hacked|hack|compromised)\b", re.IGNORECASE),
    re.compile(r"\b(safety|injury|injured|hurt|harm|danger)\b", re.IGNORECASE),
    re.compile(r"\b(legal|lawsuit|attorney|lawyer|sue|court|subpoena)\b", re.IGNORECASE),
    re.compile(r"\b(privacy|gdpr|ccpa|data.?request|delete.*data|data.*deletion)\b", re.IGNORECASE),
    # Adversarial probing for internals
    re.compile(r"\b(hidden.?prompt|system.?prompt|prompt.?inject|prompt.?leak)\b", re.IGNORECASE),
    re.compile(r"\b(credential|password|api.?key|secret.?key)\b", re.IGNORECASE),
    re.compile(r"\b(risk.?score|warehouse.?note|support.?tag|another.?customer)\b", re.IGNORECASE),
    # Adversarial probing: "reveal/expose/ignore previous instructions",
    # "what is your system prompt / hidden instructions / internal notes" etc.
    # NOTE: do NOT use bare "internal" or "what is your" — they are too broad
    # and catch legitimate queries like "What is your return policy?"
    re.compile(r"\b(reveal|expose).{0,20}\b(prompt|instruction|credential|key|note|score)\b", re.IGNORECASE),
    re.compile(r"\bignore\b.{0,20}\b(previous|above|all|prior)\b.{0,20}\b(instruction|prompt|rule)\b", re.IGNORECASE),
    re.compile(r"\bwhat is your\b.{0,30}\b(system.?prompt|hidden|internal.?note|credential|password|api.?key)\b", re.IGNORECASE),
]

# Unsupported-action reasons (for handoff_reason)
_UNSUPPORTED_REASON = (
    "Customer requested an action this system cannot perform "
    "(cancellation / refund / replacement / price adjustment / warranty approval / address change). "
    "Human agent required."
)
_SAFETY_REASON = (
    "Customer reported a safety, fraud, legal, privacy, or account-security issue, "
    "or attempted to probe internal system details. Human agent required."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_order_id(text: str) -> Optional[str]:
    """Return the first ORD-NNNN match (normalised uppercase), or None."""
    match = _ORDER_ID_RE.search(text)
    return match.group(0).upper() if match else None


def _tokenise_lower(text: str) -> set[str]:
    """Return lowercase tokens from text (alpha/numeric words)."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _is_short_followup(message: str, threshold: int = 8) -> bool:
    """Return True if the message is a short follow-up (few tokens)."""
    return len(_tokenise_lower(message)) <= threshold


def _topic_overlaps(message: str, last_topic: str) -> bool:
    """Return True if the message shares ≥1 significant word with last_topic."""
    msg_tokens = _tokenise_lower(message)
    topic_tokens = {t for t in _tokenise_lower(last_topic) if len(t) >= 4}
    return bool(msg_tokens & topic_tokens)


def _message_has_order_followup_keywords(message: str) -> bool:
    """Return True if the message contains delivery/status follow-up keywords."""
    tokens = _tokenise_lower(message)
    return bool(tokens & _ORDER_FOLLOWUP_KEYWORDS)


def _message_has_order_context_keywords(message: str) -> bool:
    """Return True if the message expresses intent to look up or track a specific order."""
    return any(pat.search(message) for pat in _ORDER_STATUS_INTENT_PATTERNS)


# ---------------------------------------------------------------------------
# Public router function
# ---------------------------------------------------------------------------

def route(session: Session, message: str) -> RouteResult:
    """
    Deterministically route *message* to a RouteDecision.

    Parameters
    ----------
    session : Session
        Current session (provides last_route, last_order_id, last_topic).
    message : str
        The raw customer message for this turn.

    Returns
    -------
    RouteResult
        Contains the decision, augmented query, and any resolved order ID.

    Notes
    -----
    * No LLM calls are made here.  All signals are regex / keyword / context.
    * The orchestrator is responsible for downgrading KNOWLEDGE_LOOKUP to
      ABSTAIN_NO_EVIDENCE when the retrieval evidence set is empty.
    """
    ctx = session.context

    # ------------------------------------------------------------------
    # Signal 1 — Safety / legal / security gate (Doc 13 rules)
    # Always escalates immediately, regardless of order ID.
    # ------------------------------------------------------------------
    for pattern in _SAFETY_ESCALATION_PATTERNS:
        if pattern.search(message):
            return RouteResult(
                decision=RouteDecision.UNSAFE_OR_UNSUPPORTED,
                query=message,
                human_handoff=True,
                handoff_reason=_SAFETY_REASON,
            )

    # ------------------------------------------------------------------
    # Signal 2 — Explicit order ID in message
    # If the user provides an order ID alongside an action request (e.g.
    # cancellation eligibility inquiry for ORD-1001), route to ORDER_LOOKUP
    # so the order can be inspected and policy cited, while flagging handoff
    # if an unsupported action was requested.
    # ------------------------------------------------------------------
    order_id = _extract_order_id(message)
    if order_id:
        has_unsupported_action = any(pat.search(message) for pat in _UNSUPPORTED_ACTION_PATTERNS)
        return RouteResult(
            decision=RouteDecision.ORDER_LOOKUP,
            query=message,
            resolved_order_id=order_id,
            human_handoff=has_unsupported_action,
            handoff_reason=_UNSUPPORTED_REASON if has_unsupported_action else None,
        )

    # ------------------------------------------------------------------
    # Signal 3 — Unsupported action without order ID (Doc 13 rules)
    # Direct requests to cancel, refund, change address, etc. without an
    # order ID cannot be fulfilled and must escalate immediately.
    # ------------------------------------------------------------------
    for pattern in _UNSUPPORTED_ACTION_PATTERNS:
        if pattern.search(message):
            return RouteResult(
                decision=RouteDecision.UNSAFE_OR_UNSUPPORTED,
                query=message,
                human_handoff=True,
                handoff_reason=_UNSUPPORTED_REASON,
            )

    # ------------------------------------------------------------------
    # Signal 4 — Multi-turn order continuation (no new ID, but we have one)
    # ------------------------------------------------------------------
    if (
        ctx.last_route == RouteDecision.ORDER_LOOKUP.value
        and ctx.last_order_id
        and _message_has_order_followup_keywords(message)
    ):
        return RouteResult(
            decision=RouteDecision.ORDER_LOOKUP,
            query=message,
            resolved_order_id=ctx.last_order_id,
        )

    # ------------------------------------------------------------------
    # Signal 5 — Order-lookup intent without an order ID
    # ------------------------------------------------------------------
    if _message_has_order_context_keywords(message):
        if not ctx.last_order_id:
            return RouteResult(
                decision=RouteDecision.NEEDS_ORDER_ID,
                query=message,
            )

    # ------------------------------------------------------------------
    # Signal 6 — Multi-turn knowledge follow-up
    # ------------------------------------------------------------------
    if ctx.last_route == RouteDecision.KNOWLEDGE_LOOKUP.value and ctx.last_topic:
        if _is_short_followup(message) or _topic_overlaps(message, ctx.last_topic):
            augmented_query = f"{ctx.last_topic} {message}".strip()
            is_damaged = any(pat.search(message) for pat in _DAMAGED_ITEM_PATTERNS)
            return RouteResult(
                decision=RouteDecision.KNOWLEDGE_LOOKUP,
                query=augmented_query,
                human_handoff=is_damaged,
            )

    # ------------------------------------------------------------------
    # Signal 7 — Default: knowledge lookup
    # ------------------------------------------------------------------
    is_damaged = any(pat.search(message) for pat in _DAMAGED_ITEM_PATTERNS)
    return RouteResult(
        decision=RouteDecision.KNOWLEDGE_LOOKUP,
        query=message,
        human_handoff=is_damaged,
    )
