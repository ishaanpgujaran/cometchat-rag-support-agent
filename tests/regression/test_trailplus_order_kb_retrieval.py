"""
tests/regression/test_trailplus_order_kb_retrieval.py
------------------------------------------------------
Regression tests for Bug 3:
  ORDER_LOOKUP route never fetched KB evidence, so queries that combined an order
  lookup with a membership-sensitive policy question (e.g. TrailPlus return window)
  always had trace.authoritative_evidence = [] — making the required_sources assertion
  on 09-trailplus-membership.md always fail.

  Root cause: orchestrator.py only fetched KB evidence for KNOWLEDGE_LOOKUP routes.
  When the route was ORDER_LOOKUP, `evidence` stayed as an empty list and
  `trace.authoritative_evidence` was never populated.

  Fix: After fetching the order result, if membership_tier == 'trailplus', run a
  supplemental KB retrieval. The resulting authoritative evidence is added to
  trace.authoritative_evidence and passed to build_messages() alongside the order data.

Regression coverage:
  test_trailplus_order_retrieval_cites_membership_doc:
    Sends "I just received order ORD-1002. What is my return window?" through the
    REAL pipeline (no mock on retrieval). ORD-1002 has membership_tier=trailplus.
    Asserts that trace.authoritative_evidence contains 09-trailplus-membership.md.

  test_standard_tier_order_no_supplemental_retrieval:
    Sends a query for ORD-1007 (standard tier). Verifies that trace.authoritative_evidence
    is still empty (no unnecessary supplemental retrieval for standard orders).
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.agent.orchestrator import handle_message_with_trace
from app.session.store import SessionStore


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


class TestTrailPlusOrderKBRetrieval:
    """Bug 3: TrailPlus order lookup must also retrieve KB evidence for policy citation."""

    def test_trailplus_order_retrieval_cites_membership_doc(self):
        """
        ORD-1002 has membership_tier=trailplus.
        Querying the return window for this order must result in
        09-trailplus-membership.md appearing in trace.authoritative_evidence.

        Before the fix: trace.authoritative_evidence was always [] for ORDER_LOOKUP,
        making required_sources always fail for this case.
        After the fix: supplemental retrieval populates the evidence and the LLM
        can cite the correct 45-day return window.
        """
        session_id = str(uuid.uuid4())
        message = "I just received order ORD-1002. What is my return window for it?"

        mock_client = _mock_gemini_client(
            "Since your order was placed while your TrailPlus membership was active, "
            "you have a 45-calendar-day return window from the date of delivery. "
            "[09-trailplus-membership.md#TrailPlus Return Window]"
        )

        with patch("app.agent.orchestrator._make_gemini_model", return_value=mock_client):
            resp, trace = handle_message_with_trace(session_id, message)

        # Core assertion: supplemental KB retrieval must have populated authoritative_evidence
        auth_filenames = {cit.split("#")[0] for cit in trace.authoritative_evidence}
        assert "09-trailplus-membership.md" in auth_filenames, (
            f"09-trailplus-membership.md not in trace.authoritative_evidence: "
            f"{trace.authoritative_evidence}. "
            "Bug 3 regression: ORDER_LOOKUP did not fetch supplemental KB evidence "
            "for trailplus membership tier."
        )

        # Tool must have been called for ORD-1002
        tool_names = [tc.name for tc in trace.tool_calls]
        assert "lookup_order" in tool_names, (
            f"lookup_order not called. tool_calls: {tool_names}"
        )
        order_ids = [tc.args.get("order_id") for tc in trace.tool_calls if tc.name == "lookup_order"]
        assert "ORD-1002" in order_ids, (
            f"ORD-1002 not in tool call args: {order_ids}"
        )

    def test_standard_tier_order_no_supplemental_retrieval(self):
        """
        ORD-1007 has membership_tier=standard.
        Querying a standard order's status must NOT trigger supplemental retrieval --
        trace.authoritative_evidence should remain [] (the order data is sufficient).
        This ensures the fix does not add unnecessary retrieval overhead for non-trailplus orders.
        """
        session_id = str(uuid.uuid4())
        message = "Where is ORD-1007 and when should it arrive?"

        mock_client = _mock_gemini_client(
            "Order ORD-1007 is in transit with UPS and is estimated to arrive on August 22, 2026."
        )

        with patch("app.agent.orchestrator._make_gemini_model", return_value=mock_client):
            resp, trace = handle_message_with_trace(session_id, message)

        # Standard tier: no supplemental retrieval → authoritative_evidence stays []
        assert trace.authoritative_evidence == [], (
            f"Expected empty authoritative_evidence for standard-tier order lookup, "
            f"but got: {trace.authoritative_evidence}. "
            "The supplemental retrieval must only fire for membership_tier=trailplus."
        )

        # Tool must still have been called
        tool_names = [tc.name for tc in trace.tool_calls]
        assert "lookup_order" in tool_names
