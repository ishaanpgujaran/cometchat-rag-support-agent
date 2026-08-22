"""
tools/verify_walkthrough.py
---------------------------
Verify the three multi-turn scenarios and edge cases end-to-end.
"""

import uuid
from app.agent.orchestrator import handle_message_with_trace

def run_scenario_1():
    print("=" * 60)
    print("SCENARIO 1: International shipping follow-up")
    print("=" * 60)
    session_id = f"test_s1_{uuid.uuid4().hex[:6]}"

    # Turn 1
    msg1 = "Do you ship internationally?"
    resp1, trace1 = handle_message_with_trace(session_id, msg1)
    print(f"User: {msg1}")
    print(f"Route: {trace1.route_decision}")
    print(f"Agent: {resp1.text}")
    print(f"Citations: {resp1.citations}")
    print(f"Handoff: {resp1.human_handoff}")
    print()

    # Turn 2
    msg2 = "What about Canada?"
    resp2, trace2 = handle_message_with_trace(session_id, msg2)
    print(f"User: {msg2}")
    print(f"Route: {trace2.route_decision}")
    print(f"Agent: {resp2.text}")
    print(f"Citations: {resp2.citations}")
    print(f"Handoff: {resp2.human_handoff}")
    print()


def run_scenario_2():
    print("=" * 60)
    print("SCENARIO 2: Order status followed by arrival")
    print("=" * 60)
    session_id = f"test_s2_{uuid.uuid4().hex[:6]}"

    # Turn 1
    msg1 = "Where is ORD-1007?"
    resp1, trace1 = handle_message_with_trace(session_id, msg1)
    print(f"User: {msg1}")
    print(f"Route: {trace1.route_decision}")
    print(f"Agent: {resp1.text}")
    print(f"Tool calls: {[tc.name for tc in trace1.tool_calls]}")
    print(f"Sanitized result: {trace1.sanitized_tool_results}")
    print(f"Handoff: {resp1.human_handoff}")
    print()

    # Turn 2
    msg2 = "When will it arrive?"
    resp2, trace2 = handle_message_with_trace(session_id, msg2)
    print(f"User: {msg2}")
    print(f"Route: {trace2.route_decision}")
    print(f"Agent: {resp2.text}")
    print(f"Tool calls: {[tc.name for tc in trace2.tool_calls]}")
    print(f"Handoff: {resp2.human_handoff}")
    print()


def run_scenario_3():
    print("=" * 60)
    print("SCENARIO 3: Return policy followed by sale items")
    print("=" * 60)
    session_id = f"test_s3_{uuid.uuid4().hex[:6]}"

    # Turn 1
    msg1 = "What is your return policy?"
    resp1, trace1 = handle_message_with_trace(session_id, msg1)
    print(f"User: {msg1}")
    print(f"Route: {trace1.route_decision}")
    print(f"Agent: {resp1.text}")
    print(f"Citations: {resp1.citations}")
    print(f"Handoff: {resp1.human_handoff}")
    print()

    # Turn 2
    msg2 = "What about sale items?"
    resp2, trace2 = handle_message_with_trace(session_id, msg2)
    print(f"User: {msg2}")
    print(f"Route: {trace2.route_decision}")
    print(f"Agent: {resp2.text}")
    print(f"Citations: {resp2.citations}")
    print(f"Handoff: {resp2.human_handoff}")
    print()


def run_scenario_unsupported():
    print("=" * 60)
    print("SCENARIO 4: Unsupported Action (Cancel Order)")
    print("=" * 60)
    session_id = f"test_s4_{uuid.uuid4().hex[:6]}"
    msg = "Please cancel my order ORD-1007 right now."
    resp, trace = handle_message_with_trace(session_id, msg)
    print(f"User: {msg}")
    print(f"Route: {trace.route_decision}")
    print(f"Agent: {resp.text}")
    print(f"Handoff: {resp.human_handoff}")
    print()


if __name__ == "__main__":
    run_scenario_1()
    run_scenario_2()
    run_scenario_3()
    run_scenario_unsupported()
