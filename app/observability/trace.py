"""
app/observability/trace.py
--------------------------
Pydantic model capturing the full execution trace for one pipeline run.

Design goals
~~~~~~~~~~~~
* Every public field is serialisable to JSON via ``model_dump_json()``.
* **Privacy-safe by construction** — the model never stores:
    - Raw KB chunk text (only ``filename#heading`` + scores)
    - Raw conflict chunk text (only filenames + metadata)
    - Fields absent from ``SafeOrderResult`` (``risk_score``, ``internal_notes``,
      ``warehouse_note``, ``support_tags``, customer email, raw address)
    - The GEMINI_API_KEY value
* All nested sub-models use only scalar / primitive types so no Chunk or
  ScoredEvidence objects are held by the Trace.
* Timestamps use ``datetime`` objects in UTC.

Sub-models
~~~~~~~~~~
``CandidateRef``  -- one retrieved chunk, scores only, no text
``ConflictRef``   -- one conflict group, filenames + metadata only
``ToolCallRef``   -- one tool invocation (name + args dict)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CandidateRef(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    filename: str
    heading: str
    document_id: str
    dense_score: float
    bm25_score: float
    final_score: float
    is_authoritative: bool
    audience: str
    status: str


class ConflictRef(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    topic: str
    doc_a_filename: str
    doc_b_filename: str
    note: str
    source: str
    confidence: str


class ToolCallRef(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    """Full execution trace for one handle_message_with_trace() call."""

    model_config = ConfigDict(use_enum_values=True)

    # Identity
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str

    # Input
    user_message: str

    # Session context snapshot (at pipeline start, before any mutation)
    session_context_used: dict[str, Any] = Field(default_factory=dict)

    # Routing
    route_decision: Optional[str] = None
    handoff_reason: Optional[str] = None

    # Retrieval (no chunk text -- filenames + scores only)
    retrieved_candidates: list[CandidateRef] = Field(default_factory=list)
    authoritative_evidence: list[str] = Field(default_factory=list)

    # Conflict detection (no chunk text -- filenames + metadata only)
    conflict_groups: list[ConflictRef] = Field(default_factory=list)

    # Tool calls
    tool_calls: list[ToolCallRef] = Field(default_factory=list)
    # SafeOrderResult.model_dump() -- already the security whitelist
    sanitized_tool_results: Optional[dict[str, Any]] = None

    # Response
    final_response: str = ""

    # Validation
    validation_failures: list[str] = Field(default_factory=list)
    fallback_or_handoff_triggered: bool = False

    # Unexpected exceptions
    errors: list[str] = Field(default_factory=list)

    # Timestamps per stage (UTC, ISO-8601 on serialisation)
    ts_start: datetime = Field(default_factory=_utcnow)
    ts_routed: Optional[datetime] = None
    ts_retrieved: Optional[datetime] = None
    ts_conflicts_detected: Optional[datetime] = None
    ts_gemini_called: Optional[datetime] = None
    ts_validated: Optional[datetime] = None
    ts_end: Optional[datetime] = None
