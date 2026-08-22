"""
app/agent/orchestrator.py
--------------------------
Core orchestration loop for the support agent.

``handle_message(session_id, message) -> AgentResponse`` is the single public
entry-point.  It wires together:

  session store  →  router  →  retrieval / order lookup  →  prompt builder
  →  Gemini API (with function-calling)  →  safety validation
  →  session update  →  AgentResponse

``handle_message_with_trace(session_id, message) -> tuple[AgentResponse, Trace]``
is the traced variant.  It returns the same AgentResponse PLUS a fully-populated
``Trace`` object (see app/observability/trace.py).  All Phase 2/3 internals are
unchanged — trace collection happens exclusively at the orchestrator boundary.

Doc 13 filter
~~~~~~~~~~~~~
All evidence with ``chunk.metadata.audience == "internal"`` is filtered out
before building the evidence pack.  This ensures ``13-support-escalation.md``
and ``14-internal-content-migration-notes.md`` can never surface as citations
or grounding text.  The rules from Doc 13 are encoded as deterministic logic
in ``app/safety/trust.py`` and ``app/agent/router.py``.

Gemini function-calling
~~~~~~~~~~~~~~~~~~~~~~~
The ``lookup_order`` function is declared as a Gemini Tool.  If the model
emits a function call during generation, the orchestrator:
  1. Executes ``lookup_order()`` from ``app.orders.lookup``.
  2. Returns **only** the ``SafeOrderResult`` (never the raw order dict).
  3. Re-submits to the model with the tool response.
This loop runs at most once per turn to avoid runaway tool calls.

AgentResponse
~~~~~~~~~~~~~
  text          : cleaned, validated response text
  citations     : list of "filename#heading" strings from authoritative evidence
  human_handoff : True when escalation is required
  route         : RouteDecision.value string for observability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from google import genai
from google.genai import types as genai_types

from app.agent.prompt import SYSTEM_INSTRUCTION, build_messages
from app.agent.router import RouteDecision, RouteResult, route as do_route
from app.config import GEMINI_MODEL, require_api_key
from app.observability.logging_config import get_json_logger
from app.observability.trace import (
    CandidateRef,
    ConflictRef,
    Trace,
    ToolCallRef,
    _utcnow,
)
from app.orders.lookup import lookup_order
from app.orders.models import SafeOrderResult
from app.policy.conflict import ConflictDetector
from app.policy.scoring import ScoredEvidence, is_authoritative, score_and_rank
from app.retrieval.index import hybrid_search
from app.safety.trust import validate_response
from app.session.store import (
    Session,
    SessionStore,
    _store as _session_store,
    append_turn,
    get_session,
    update_context,
)

logger = logging.getLogger(__name__)
_json_logger = get_json_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RETRIEVAL_K: int = 8          # chunks fetched from hybrid search
_MIN_AUTHORITATIVE: int = 1    # minimum authoritative chunks for non-abstain
_INTERNAL_AUDIENCE: str = "internal"

# ---------------------------------------------------------------------------
# Gemini Tool declaration for lookup_order (google-genai SDK)
# ---------------------------------------------------------------------------

_LOOKUP_ORDER_TOOL = genai_types.Tool(
    function_declarations=[
        genai_types.FunctionDeclaration(
            name="lookup_order",
            description=(
                "Look up a customer order by its order ID (format: ORD-NNNN). "
                "Returns customer-safe order information including status, "
                "items, carrier, and estimated delivery. "
                "Call this whenever the customer provides a valid order ID."
            ),
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "order_id": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description="The order ID to look up, e.g. 'ORD-1007'.",
                    )
                },
                required=["order_id"],
            ),
        )
    ]
)

# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    """The final response returned by handle_message().

    Attributes
    ----------
    text : str
        The cleaned, validated response text to show the customer.
    citations : list[str]
        Source citations from authoritative evidence, formatted as
        "filename#heading" (may be empty for order-only or refusal responses).
    human_handoff : bool
        True when a human agent should be involved.
    route : str
        The RouteDecision value string (for observability / logging).
    """
    text: str
    citations: list[str] = field(default_factory=list)
    human_handoff: bool = False
    route: str = RouteDecision.KNOWLEDGE_LOOKUP.value


# ---------------------------------------------------------------------------
# Deterministic immediate responses (no LLM call)
# ---------------------------------------------------------------------------

_NEEDS_ORDER_ID_TEXT = (
    "I'd be happy to help with your order! "
    "Could you please share your order ID? "
    "It should look like **ORD-XXXX** and can be found in your confirmation email."
)

_UNSUPPORTED_ACTION_TEXT = (
    "I'm sorry, but I'm not able to process that request directly. "
    "Actions like cancellations, refunds, replacements, address changes, "
    "and warranty approvals require assistance from our support team. "
    "A human agent will be best placed to help you with this — "
    "please contact us through our support channels."
)

_ABSTAIN_TEXT = (
    "I don't have enough information in our knowledge base to answer that question reliably. "
    "I'd recommend reaching out to our support team who can look into this for you."
)


# ---------------------------------------------------------------------------
# Gemini client factory (lazy, so tests can monkeypatch before first call)
# ---------------------------------------------------------------------------

def _make_gemini_model(system_instruction: str) -> genai.Client:
    """Create and return a configured google-genai Client."""
    api_key = require_api_key()
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Evidence pipeline helpers
# ---------------------------------------------------------------------------

def _fetch_and_filter_evidence(query: str) -> list[ScoredEvidence]:
    """
    Run hybrid search + scoring, then filter out internal-audience chunks.

    This is the Doc 13 filter: 13-support-escalation.md and
    14-internal-content-migration-notes.md both have audience=internal
    and are silently excluded here.
    """
    candidates = hybrid_search(query, k=_RETRIEVAL_K)
    scored = score_and_rank(candidates)
    # Exclude internal-audience documents (Doc 13 filter)
    return [e for e in scored if e.chunk.metadata.audience != _INTERNAL_AUDIENCE]


def _extract_citations(evidence: list[ScoredEvidence]) -> list[str]:
    """Return citation strings for all authoritative evidence chunks."""
    citations: list[str] = []
    for ev in evidence:
        if is_authoritative(ev):
            m = ev.chunk.metadata
            if m.heading:
                citations.append(f"{m.filename}#{m.heading}")
            else:
                citations.append(m.filename)
    return citations


# ---------------------------------------------------------------------------
# Gemini call with optional tool-call loop
# ---------------------------------------------------------------------------

def _call_gemini_with_tools(
    model: genai.Client,
    messages: list[dict],
    system_instruction: str,
) -> tuple[str, Optional[SafeOrderResult]]:
    """
    Call Gemini (google-genai SDK), handle an optional lookup_order tool call.

    Returns
    -------
    text : str
        The final model text response.
    tool_result : SafeOrderResult | None
        The SafeOrderResult if lookup_order was called; None otherwise.
    """
    tool_result_obj: Optional[SafeOrderResult] = None

    config = genai_types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[_LOOKUP_ORDER_TOOL],
    )

    response = model.models.generate_content(
        model=GEMINI_MODEL,
        contents=messages,
        config=config,
    )

    # Check for function call in the response
    candidate = response.candidates[0] if response.candidates else None
    if candidate is None:
        return ("", None)

    # Walk parts looking for function calls
    for part in candidate.content.parts:
        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            if fc.name == "lookup_order":
                raw_order_id = dict(fc.args).get("order_id", "")
                logger.info("Model requested lookup_order('%s')", raw_order_id)
                tool_result_obj = lookup_order(raw_order_id)

                # Build follow-up conversation with tool response
                follow_up = list(messages) + [
                    genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(
                            function_call=genai_types.FunctionCall(
                                name=fc.name,
                                args=dict(fc.args),
                            )
                        )],
                    ),
                    genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name=fc.name,
                                response=tool_result_obj.model_dump(exclude_none=True),
                            )
                        )],
                    ),
                ]
                response = model.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=follow_up,
                    config=config,
                )
                break  # only one tool call per turn

    # Extract text from the final response
    text = ""
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text += part.text

    return (text, tool_result_obj)


# ---------------------------------------------------------------------------
# Trace population helpers
# ---------------------------------------------------------------------------

def _build_candidate_refs(evidence: list[ScoredEvidence]) -> list[CandidateRef]:
    """Convert scored evidence into privacy-safe CandidateRef objects (no text)."""
    refs: list[CandidateRef] = []
    for ev in evidence:
        m = ev.chunk.metadata
        refs.append(CandidateRef(
            filename=m.filename,
            heading=m.heading,
            document_id=m.document_id,
            dense_score=ev.dense_score,
            bm25_score=ev.bm25_score,
            final_score=ev.final_score,
            is_authoritative=is_authoritative(ev),
            audience=m.audience,
            status=m.status,
        ))
    return refs


def _build_conflict_refs(conflict_groups: list) -> list[ConflictRef]:
    """Convert ConflictGroup objects into privacy-safe ConflictRef objects (no text)."""
    refs: list[ConflictRef] = []
    for cg in conflict_groups:
        refs.append(ConflictRef(
            topic=cg.topic,
            doc_a_filename=cg.doc_a_filename,
            doc_b_filename=cg.doc_b_filename,
            note=cg.note,
            source=cg.source,
            confidence=cg.confidence,
        ))
    return refs


# ---------------------------------------------------------------------------
# Structured log helpers
# ---------------------------------------------------------------------------

def _log_stage(
    trace_id: str,
    session_id: str,
    stage: str,
    message: str,
    level: int = logging.INFO,
    **extra_fields,
) -> None:
    """Emit one structured JSON log line for a pipeline stage.

    Forbidden fields are never passed here — callers must only supply
    safe, whitelisted fields. Full KB chunk text must never appear in
    ``extra_fields``; use filename#heading references instead.
    """
    _json_logger.log(
        level,
        message,
        extra={
            "trace_id": trace_id,
            "session_id": session_id,
            "stage": stage,
            **extra_fields,
        },
    )


# ---------------------------------------------------------------------------
# Traced entry point (primary implementation)
# ---------------------------------------------------------------------------

def handle_message_with_trace(
    session_id: str,
    message: str,
) -> tuple[AgentResponse, Trace]:
    """
    Handle a single customer message and return both the AgentResponse and a
    fully-populated Trace.

    This is the primary implementation.  ``handle_message()`` calls this
    function and discards the Trace for backward compatibility.

    Full pipeline
    -------------
    1.  Load session
    2.  Route message (deterministic, no LLM)
    3.  Immediate deterministic response for UNSAFE / NEEDS_ORDER_ID
    4.  Evidence fetch + filter (KNOWLEDGE_LOOKUP)
    5.  Order lookup (ORDER_LOOKUP)
    6.  Detect conflicts
    7.  Build Gemini messages
    8.  Call Gemini (with function-calling)
    9.  Validate and clean response (deterministic)
    10. Update session
    11. Return AgentResponse + Trace
    """
    trace = Trace(session_id=session_id, user_message=message)

    try:
        session: Session = _session_store.get_session(session_id)

        # Snapshot context BEFORE any mutation
        ctx = session.context
        trace.session_context_used = {
            "last_order_id": ctx.last_order_id,
            "last_topic": ctx.last_topic,
            "last_route": ctx.last_route,
        }

        _log_stage(
            trace.trace_id, session_id, "start",
            "pipeline=started",
            route_decision=None,
        )

        # --------------------------------------------------------------
        # 2. Route
        # --------------------------------------------------------------
        route_result: RouteResult = do_route(session, message)
        decision = route_result.decision
        trace.ts_routed = _utcnow()
        trace.route_decision = decision.value
        trace.handoff_reason = route_result.handoff_reason

        logger.info(
            "[session=%s] route=%s query='%s'",
            session_id, decision.value, route_result.query,
        )
        _log_stage(
            trace.trace_id, session_id, "route",
            "stage=routed",
            route_decision=decision.value,
            query_length=len(route_result.query),
            human_handoff=route_result.human_handoff,
            handoff_reason=route_result.handoff_reason,
        )

        # --------------------------------------------------------------
        # 3. Immediate deterministic responses (no LLM call needed)
        # --------------------------------------------------------------
        if decision == RouteDecision.UNSAFE_OR_UNSUPPORTED:
            trace.final_response = _UNSUPPORTED_ACTION_TEXT
            trace.fallback_or_handoff_triggered = True
            trace.ts_end = _utcnow()
            _log_stage(
                trace.trace_id, session_id, "done",
                "stage=done route=UNSAFE_OR_UNSUPPORTED",
                fallback_triggered=True,
            )
            _commit_turn(session_id, message, _UNSUPPORTED_ACTION_TEXT, route_result)
            return (
                AgentResponse(
                    text=_UNSUPPORTED_ACTION_TEXT,
                    human_handoff=True,
                    route=decision.value,
                ),
                trace,
            )

        if decision == RouteDecision.NEEDS_ORDER_ID:
            trace.final_response = _NEEDS_ORDER_ID_TEXT
            trace.fallback_or_handoff_triggered = False
            trace.ts_end = _utcnow()
            _log_stage(
                trace.trace_id, session_id, "done",
                "stage=done route=NEEDS_ORDER_ID",
                fallback_triggered=False,
            )
            _commit_turn(session_id, message, _NEEDS_ORDER_ID_TEXT, route_result)
            return (
                AgentResponse(
                    text=_NEEDS_ORDER_ID_TEXT,
                    human_handoff=False,
                    route=decision.value,
                ),
                trace,
            )

        # --------------------------------------------------------------
        # 4. Evidence pipeline (KNOWLEDGE_LOOKUP)
        # --------------------------------------------------------------
        evidence: list[ScoredEvidence] = []
        conflict_groups = []

        if decision == RouteDecision.KNOWLEDGE_LOOKUP:
            evidence = _fetch_and_filter_evidence(route_result.query)
            trace.ts_retrieved = _utcnow()
            trace.retrieved_candidates = _build_candidate_refs(evidence)
            trace.authoritative_evidence = _extract_citations(evidence)

            auth_count = sum(1 for e in evidence if is_authoritative(e))
            _log_stage(
                trace.trace_id, session_id, "retrieve",
                "stage=retrieved",
                candidate_count=len(evidence),
                authoritative_count=auth_count,
                # Log filename#heading refs only -- never chunk text
                top_refs=[
                    f"{e.chunk.metadata.filename}#{e.chunk.metadata.heading}"
                    for e in evidence[:3]
                ],
            )

            conflict_groups = ConflictDetector().detect(evidence)
            trace.ts_conflicts_detected = _utcnow()
            trace.conflict_groups = _build_conflict_refs(conflict_groups)
            _log_stage(
                trace.trace_id, session_id, "detect_conflicts",
                "stage=conflicts_detected",
                conflict_count=len(conflict_groups),
                conflicts=[
                    {"topic": cg.topic, "confidence": cg.confidence}
                    for cg in conflict_groups
                ],
            )

            # Downgrade to ABSTAIN if no authoritative evidence found
            authoritative_count = sum(1 for e in evidence if is_authoritative(e))
            if authoritative_count < _MIN_AUTHORITATIVE:
                logger.info("[session=%s] No authoritative evidence — abstaining.", session_id)
                decision = RouteDecision.ABSTAIN_NO_EVIDENCE
                trace.route_decision = decision.value
                trace.final_response = _ABSTAIN_TEXT
                trace.fallback_or_handoff_triggered = True
                trace.ts_end = _utcnow()
                _log_stage(
                    trace.trace_id, session_id, "done",
                    "stage=done route=ABSTAIN_NO_EVIDENCE",
                    fallback_triggered=True,
                )
                _commit_turn(
                    session_id, message, _ABSTAIN_TEXT, route_result,
                    override_route=decision,
                )
                return (
                    AgentResponse(
                        text=_ABSTAIN_TEXT,
                        human_handoff=True,
                        route=decision.value,
                    ),
                    trace,
                )

        # --------------------------------------------------------------
        # 5. Order lookup (ORDER_LOOKUP)
        # --------------------------------------------------------------
        pre_fetched_tool_result: Optional[SafeOrderResult] = None
        if decision == RouteDecision.ORDER_LOOKUP:
            order_id = route_result.resolved_order_id or ""
            trace.tool_calls.append(ToolCallRef(
                name="lookup_order",
                args={"order_id": order_id},
            ))
            pre_fetched_tool_result = lookup_order(order_id)
            logger.info(
                "[session=%s] lookup_order('%s') → found=%s status=%s",
                session_id, order_id,
                pre_fetched_tool_result.found,
                pre_fetched_tool_result.status,
            )
            _log_stage(
                trace.trace_id, session_id, "tool_call",
                "stage=tool_call name=lookup_order",
                tool_name="lookup_order",
                order_id=order_id,
                found=pre_fetched_tool_result.found,
                status=pre_fetched_tool_result.status,
            )

        # --------------------------------------------------------------
        # 6. Append the current user turn to session
        # --------------------------------------------------------------
        _session_store.append_turn(session_id, "user", message)

        # --------------------------------------------------------------
        # 7. Build prompt messages
        # --------------------------------------------------------------
        messages = build_messages(
            session=_session_store.get_session(session_id),
            route=decision,
            evidence=evidence if decision == RouteDecision.KNOWLEDGE_LOOKUP else [],
            tool_result=pre_fetched_tool_result,
        )

        # --------------------------------------------------------------
        # 8. Gemini call
        # --------------------------------------------------------------
        model = _make_gemini_model(SYSTEM_INSTRUCTION)
        raw_text, tool_result_from_model = _call_gemini_with_tools(
            model, messages, SYSTEM_INSTRUCTION
        )
        trace.ts_gemini_called = _utcnow()

        # Record model-triggered tool call if any (not already recorded above)
        if tool_result_from_model is not None and pre_fetched_tool_result is None:
            trace.tool_calls.append(ToolCallRef(
                name="lookup_order",
                args={"order_id": tool_result_from_model.order_id},
            ))

        # Prefer the pre-fetched tool result; use model-triggered one only if needed.
        effective_tool_result = pre_fetched_tool_result or tool_result_from_model

        # Capture sanitized tool result (SafeOrderResult whitelist -- no raw fields)
        if effective_tool_result is not None:
            trace.sanitized_tool_results = effective_tool_result.model_dump(exclude_none=True)

        _log_stage(
            trace.trace_id, session_id, "gemini_call",
            "stage=gemini_returned",
            response_length=len(raw_text),
            tool_called=effective_tool_result is not None,
        )

        # --------------------------------------------------------------
        # 9. Validate and clean
        # --------------------------------------------------------------
        validation = validate_response(
            raw_model_output=raw_text,
            evidence_pack=evidence,
            tool_result=effective_tool_result,
            conflict_groups=conflict_groups,
            force_handoff=route_result.human_handoff,
            handoff_reason=route_result.handoff_reason,
        )
        trace.ts_validated = _utcnow()
        trace.validation_failures = list(validation.flags)
        trace.fallback_or_handoff_triggered = validation.human_handoff

        if validation.flags:
            logger.warning(
                "[session=%s] Safety flags: %s", session_id, "; ".join(validation.flags)
            )
            _log_stage(
                trace.trace_id, session_id, "validate",
                "stage=validated flags_raised=true",
                level=logging.WARNING,
                flag_count=len(validation.flags),
                flags=validation.flags,
                human_handoff=validation.human_handoff,
            )
        else:
            _log_stage(
                trace.trace_id, session_id, "validate",
                "stage=validated flags_raised=false",
                flag_count=0,
                human_handoff=validation.human_handoff,
            )

        # --------------------------------------------------------------
        # 10. Build citations from authoritative evidence
        # --------------------------------------------------------------
        citations = _extract_citations(evidence)

        # --------------------------------------------------------------
        # 11. Update session (assistant turn + context)
        # --------------------------------------------------------------
        _session_store.append_turn(session_id, "assistant", validation.cleaned_response)

        # Determine last_topic for multi-turn context
        last_topic: Optional[str] = None
        if evidence:
            for ev in evidence:
                if is_authoritative(ev) and ev.chunk.metadata.heading:
                    last_topic = ev.chunk.metadata.heading
                    break

        _session_store.update_context(
            session_id,
            last_order_id=route_result.resolved_order_id or session.context.last_order_id,
            last_topic=last_topic,
            last_route=decision.value,
        )

        # Populate final trace fields
        trace.final_response = validation.cleaned_response
        trace.ts_end = _utcnow()

        _log_stage(
            trace.trace_id, session_id, "done",
            "stage=done",
            route_decision=decision.value,
            citation_count=len(citations),
            human_handoff=validation.human_handoff,
            fallback_triggered=validation.human_handoff,
        )

        return (
            AgentResponse(
                text=validation.cleaned_response,
                citations=citations,
                human_handoff=validation.human_handoff,
                route=decision.value,
            ),
            trace,
        )

    except Exception as exc:
        trace.errors.append(str(exc))
        trace.ts_end = _utcnow()
        _log_stage(
            trace.trace_id, session_id, "error",
            f"stage=error exception={type(exc).__name__}",
            level=logging.ERROR,
            exception_type=type(exc).__name__,
        )
        raise


# ---------------------------------------------------------------------------
# Backward-compatible public entry point
# ---------------------------------------------------------------------------

def handle_message(session_id: str, message: str) -> AgentResponse:
    """
    Handle a single customer message and return an AgentResponse.

    Full pipeline
    -------------
    1. Load session
    2. Route message (deterministic, no LLM)
    3. Immediate deterministic response for UNSAFE / NEEDS_ORDER_ID
    4. Evidence fetch + filter (KNOWLEDGE_LOOKUP)
    5. Order lookup (ORDER_LOOKUP)
    6. Detect conflicts
    7. Build Gemini messages
    8. Call Gemini (with function-calling)
    9. Validate and clean response (deterministic)
    10. Update session
    11. Return AgentResponse

    Note: This function calls handle_message_with_trace() internally and
    discards the Trace for backward compatibility.  Use
    handle_message_with_trace() directly when observability data is needed.
    """
    response, _trace = handle_message_with_trace(session_id, message)
    return response


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _commit_turn(
    session_id: str,
    user_message: str,
    assistant_response: str,
    route_result: RouteResult,
    override_route: Optional[RouteDecision] = None,
) -> None:
    """Append both turns and update context for deterministic (no-LLM) paths."""
    _session_store.append_turn(session_id, "user", user_message)
    _session_store.append_turn(session_id, "assistant", assistant_response)
    _session_store.update_context(
        session_id,
        last_order_id=route_result.resolved_order_id,
        last_route=(override_route or route_result.decision).value,
    )
