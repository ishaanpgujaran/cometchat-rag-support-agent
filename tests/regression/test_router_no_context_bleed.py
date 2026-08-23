"""
tests/regression/test_router_no_context_bleed.py
-------------------------------------------------
Regression test: ensure session.context.last_order_id is only reused when
incoming messages contain order continuation signals, preventing context bleed
into unrelated policy inquiries.
"""

import pytest

from app.agent.router import RouteDecision, route
from app.session.store import Session, SessionContext


def test_router_no_context_bleed_after_order_lookup():
    # Setup session with last_order_id="ORD-1007" and last_route=ORDER_LOOKUP
    session = Session(session_id="test-context-bleed")
    session.context = SessionContext(
        last_order_id="ORD-1007",
        last_route=RouteDecision.ORDER_LOOKUP.value,
        last_topic="Where is my order ORD-1007?",
    )

    # 1. Unrelated international shipping question
    res1 = route(session, "Do you ship internationally?")
    assert res1.decision == RouteDecision.KNOWLEDGE_LOOKUP
    assert res1.resolved_order_id is None

    # 2. Unrelated warranty policy question
    res2 = route(session, "What is the warranty on bags?")
    assert res2.decision == RouteDecision.KNOWLEDGE_LOOKUP
    assert res2.resolved_order_id is None

    # 3. Order-continuation follow-up question
    res3 = route(session, "When will it arrive?")
    assert res3.decision == RouteDecision.ORDER_LOOKUP
    assert res3.resolved_order_id == "ORD-1007"
