"""
tools/smoke_test.py
--------------------
Manual smoke test — uses a REAL Gemini API call.
Requires GEMINI_API_KEY set in .env or the environment.

Run with:
    python tools/smoke_test.py

NOT part of the automated test suite (no pytest markers).
Demonstrates one full knowledge question and one full order question.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

# Ensure the project root is on sys.path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.agent.orchestrator import handle_message  # noqa: E402


def _hr(title: str = "") -> None:
    width = 70
    if title:
        side = (width - len(title) - 2) // 2
        print("=" * side + f" {title} " + "=" * side)
    else:
        print("=" * width)


def _print_response(label: str, resp) -> None:
    _hr(label)
    print(f"Route         : {resp.route}")
    print(f"Human Handoff : {resp.human_handoff}")
    print(f"Citations     : {resp.citations or '(none)'}")
    print()
    print("Response text:")
    print(textwrap.fill(resp.text, width=70))
    _hr()
    print()


def main() -> None:
    _hr("CometChat RAG Support Agent — Smoke Test")
    print()

    # ------------------------------------------------------------------
    # Test A — Knowledge question: international shipping
    # ------------------------------------------------------------------
    print("Sending knowledge question: 'Do you ship to Canada?'")
    session_a = "smoke-knowledge-001"
    resp_a = handle_message(session_a, "Do you ship to Canada?")
    _print_response("Knowledge Question Result", resp_a)

    # ------------------------------------------------------------------
    # Test B — Follow-up knowledge question (multi-turn context)
    # ------------------------------------------------------------------
    print("Sending follow-up: 'What about the UK?'")
    resp_b = handle_message(session_a, "What about the UK?")
    _print_response("Follow-up Result (should use prior topic)", resp_b)

    # ------------------------------------------------------------------
    # Test C — Order question
    # ------------------------------------------------------------------
    print("Sending order question: 'What is the status of order ORD-1007?'")
    session_b = "smoke-order-001"
    resp_c = handle_message(session_b, "What is the status of order ORD-1007?")
    _print_response("Order Question Result", resp_c)

    # ------------------------------------------------------------------
    # Test D — Cancel/refund (should be refused, no LLM call)
    # ------------------------------------------------------------------
    print("Sending refund request (should be immediately refused):")
    session_c = "smoke-refund-001"
    resp_d = handle_message(session_c, "I want a refund for my last order")
    _print_response("Refund Request Result", resp_d)

    # Summary
    _hr("Summary")
    results = [
        ("Knowledge question", resp_a.route == "KNOWLEDGE_LOOKUP" and bool(resp_a.citations)),
        ("Follow-up context routing", resp_b.route == "KNOWLEDGE_LOOKUP"),
        ("Order lookup", resp_c.route == "ORDER_LOOKUP" and resp_c.human_handoff is False),
        ("Refund refusal", resp_d.route == "UNSAFE_OR_UNSUPPORTED" and resp_d.human_handoff),
    ]
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False
    print()
    if all_passed:
        print("All smoke tests passed.")
    else:
        print("Some smoke tests failed — see output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
