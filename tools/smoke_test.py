"""
tools/smoke_test.py
--------------------
Manual / integration smoke test — uses a REAL Gemini API call.
Requires GEMINI_API_KEY set in .env or the environment.

Run with:
    python tools/smoke_test.py

Sends one real request ("What are return policies") through
app.agent.orchestrator.handle_message and asserts a non-empty answer
with at least one citation comes back.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.agent.orchestrator import handle_message  # noqa: E402


def main() -> None:
    session_id = "smoke-test-session"
    question = "What are return policies"
    print(f"Sending real request: '{question}'...")

    response = handle_message(session_id, question)

    print("=" * 60)
    print(f"Route         : {response.route}")
    print(f"Human Handoff : {response.human_handoff}")
    print(f"Citations     : {response.citations}")
    print("Response text :")
    print(response.text)
    print("=" * 60)

    # Assertions
    assert response.text and len(response.text.strip()) > 0, "Response text must not be empty"
    assert len(response.citations) >= 1, f"Expected at least one citation, got {response.citations}"
    print("✅ Smoke test passed!")


if __name__ == "__main__":
    main()

