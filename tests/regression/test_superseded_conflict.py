"""
tests/regression/test_superseded_conflict.py
---------------------------------------------
Regression tests for:
1. Superseded documents leaking into authoritative evidence and conflict detection.
   - test_superseded_returns_policy_not_flagged_as_conflict: query about return windows
     must NOT trigger human_handoff via conflict, must cite ONLY
     01-returns-policy-current.md, must NEVER cite or quote 02-returns-policy-legacy.md as
     authoritative, and must state 30 calendar days (not 45, not "it depends," not "sources
     conflict").
2. Genuine conflict detection preservation.
   - test_genuine_tumbler_conflict_still_detected: confirm the Breeze Tumbler
     dishwasher-safety case (11 vs 12) still correctly triggers human_handoff=True and cites
     both — this proves the fix didn't overcorrect into suppressing all conflict detection.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch
import pytest

from app.agent.orchestrator import handle_message, handle_message_with_trace
from app.agent.router import RouteDecision
from app.ingestion.models import Chunk, ChunkMetadata
from app.policy.conflict import ConflictDetector
from app.policy.scoring import ScoredEvidence, filter_authoritative
from app.session.store import SessionStore


def _make_chunk(
    filename: str,
    heading: str,
    text: str,
    status: str = "active",
    audience: str = "customer",
    policy_authority: str = "official",
    customer_answering: bool = True,
    document_id: str = "DOC-TEST",
    title: str = "Test Doc",
    supersedes: str | None = None,
    superseded_by: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"{filename}__{heading.lower().replace(' ', '_')}",
        text=text,
        metadata=ChunkMetadata(
            filename=filename,
            document_id=document_id,
            title=title,
            status=status,
            audience=audience,
            policy_authority=policy_authority,
            heading=heading,
            customer_answering=customer_answering,
            supersedes=supersedes,
            superseded_by=superseded_by,
        ),
    )


def _make_evidence(
    filename: str,
    heading: str,
    text: str,
    final_score: float = 0.85,
    **chunk_kwargs,
) -> ScoredEvidence:
    chunk = _make_chunk(filename, heading, text, **chunk_kwargs)
    return ScoredEvidence(
        chunk=chunk,
        bm25_score=0.5,
        dense_score=0.7,
        metadata_bonus=0.25,
        metadata_penalty=0.0,
        final_score=final_score,
    )


def _mock_gemini_client(text: str) -> MagicMock:
    part = MagicMock()
    part.text = text
    part.function_call = None

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    response.text = text

    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


@pytest.fixture(autouse=True)
def isolate_sessions():
    fresh_store = SessionStore()
    with patch("app.session.store._store", fresh_store), \
         patch("app.agent.orchestrator._session_store", fresh_store):
        yield fresh_store


def test_superseded_returns_policy_not_flagged_as_conflict():
    """
    Query about return windows must NOT trigger human_handoff via conflict,
    must cite ONLY 01-returns-policy-current.md, must NEVER cite or quote
    02-returns-policy-legacy.md as authoritative, and must state 30 calendar days
    (not 45, not "it depends", not "sources conflict").
    """
    session_id = f"test-return-{uuid.uuid4().hex}"

    # Document 01: active, official, 30 days
    ev_01 = _make_evidence(
        filename="01-returns-policy-current.md",
        heading="Standard return window",
        text="Standard plan customers may request a return within 30 calendar days of delivery for eligible items.",
        document_id="RET-2026-01",
        title="Returns Policy (Current)",
        status="active",
        audience="customer",
        policy_authority="official",
        supersedes="RET-2024-01",
        final_score=0.95,
    )

    # Document 02: superseded, official, 45 days
    ev_02 = _make_evidence(
        filename="02-returns-policy-legacy.md",
        heading="Return window",
        text="Eligible merchandise could be returned within 45 calendar days of delivery.",
        document_id="RET-2024-01",
        title="Returns Policy (Legacy)",
        status="superseded",
        audience="customer",
        policy_authority="official",
        superseded_by="RET-2026-01",
        final_score=0.40,
    )

    raw_evidence = [ev_01, ev_02]

    # Verify filter_authoritative removes the superseded document
    authoritative = filter_authoritative(raw_evidence)
    assert len(authoritative) == 1
    assert authoritative[0].chunk.metadata.filename == "01-returns-policy-current.md"

    # Verify ConflictDetector does not flag conflict between 01 and 02
    detector = ConflictDetector()
    assert detector.detect(raw_evidence) == []
    assert detector.detect(authoritative) == []

    # Mock Gemini model response
    mock_client = _mock_gemini_client(
        "Standard customers may return items within 30 calendar days of delivery. "
        "[01-returns-policy-current.md#Standard return window]"
    )

    with patch("app.agent.orchestrator._make_gemini_model", return_value=mock_client), \
         patch("app.agent.orchestrator._fetch_and_filter_evidence", return_value=raw_evidence):
        resp, trace = handle_message_with_trace(session_id, "How long do I have to return an item?")

    # Assertions
    assert resp.human_handoff is False, "Superseded doc must not trigger human handoff"
    assert resp.route == RouteDecision.KNOWLEDGE_LOOKUP.value
    assert len(resp.citations) == 1
    assert "01-returns-policy-current.md#Standard return window" in resp.citations
    assert not any("02-returns-policy-legacy.md" in c for c in resp.citations), (
        "02-returns-policy-legacy.md must NEVER be cited as authoritative"
    )

    assert "30" in resp.text
    assert "45" not in resp.text
    assert "conflict" not in resp.text.lower()
    assert "it depends" not in resp.text.lower()
    assert trace.conflict_groups == []


def test_genuine_tumbler_conflict_still_detected():
    """
    Confirm the Breeze Tumbler dishwasher-safety case (11 vs 12) still
    correctly triggers human_handoff=True and cites both — proving the fix
    did not suppress genuine conflict detection.
    """
    session_id = f"test-conflict-{uuid.uuid4().hex}"

    ev_11 = _make_evidence(
        filename="11-product-care.md",
        heading="Cleaning Instructions",
        text="The Breeze Tumbler body must be hand-washed only. The lid may go on the top rack.",
        document_id="CARE-2026-01",
        title="Product Care Guide",
        status="active",
        audience="customer",
        policy_authority="official",
        final_score=0.90,
    )
    ev_12 = _make_evidence(
        filename="12-breeze-tumbler-product-card.md",
        heading="Care Instructions",
        text="All Breeze Tumbler components are dishwasher safe.",
        document_id="PROD-2026-12",
        title="Breeze Tumbler Product Card",
        status="active",
        audience="customer",
        policy_authority="official",
        final_score=0.88,
    )

    evidence = [ev_11, ev_12]

    # Verify filter_authoritative keeps both active official chunks
    authoritative = filter_authoritative(evidence)
    assert len(authoritative) == 2

    # Verify ConflictDetector flags the genuine conflict
    detector = ConflictDetector()
    conflicts = detector.detect(authoritative)
    assert len(conflicts) == 1
    assert conflicts[0].topic == "breeze_tumbler_dishwasher_safety"

    # Model generates text
    mock_client = _mock_gemini_client(
        "There are conflicting instructions regarding dishwasher safety for the Breeze Tumbler."
    )

    with patch("app.agent.orchestrator._make_gemini_model", return_value=mock_client), \
         patch("app.agent.orchestrator._fetch_and_filter_evidence", return_value=evidence):
        resp, trace = handle_message_with_trace(session_id, "Can I put the Breeze Tumbler in the dishwasher?")

    assert resp.human_handoff is True, "Genuine conflict must trigger human handoff"
    assert "11-product-care.md" in resp.text
    assert "12-breeze-tumbler-product-card.md" in resp.text
    assert len(trace.conflict_groups) == 1
