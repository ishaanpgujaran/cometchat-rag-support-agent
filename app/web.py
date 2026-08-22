"""
app/web.py
----------
Streamlit chat interface for Aster & Row support agent.

Maintains session isolation in st.session_state, displays citations and human
handoff banners, and includes a sidebar debug panel for inspecting traces.
Contains NO business logic — purely wraps handle_message_with_trace().
"""

from __future__ import annotations

import sys
from pathlib import Path
import uuid

# Ensure project root is in sys.path when launched via `streamlit run app/web.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.agent.orchestrator import handle_message_with_trace
from app.observability.trace import Trace

HUMAN_HANDOFF_MESSAGE = (
    "To give you the most accurate help with this, "
    "I’m going to get a human expert on the line for you."
)


def _render_debug_sidebar(trace: Trace) -> None:
    """Render the execution trace in the sidebar."""
    st.sidebar.markdown("### 🔍 Execution Trace")
    st.sidebar.markdown(f"**Route Decision:** `{trace.route_decision}`")

    if trace.handoff_reason:
        st.sidebar.markdown(f"**Handoff Reason:** `{trace.handoff_reason}`")

    if trace.fallback_or_handoff_triggered:
        st.sidebar.warning("⚠️ Fallback / Handoff Triggered")

    # Retrieved candidates
    st.sidebar.markdown("#### Retrieved Candidates")
    if trace.retrieved_candidates:
        records = [
            {
                "Source": f"{c.filename}#{c.heading}" if c.heading else c.filename,
                "Final": round(c.final_score, 3),
                "Dense": round(c.dense_score, 3),
                "BM25": round(c.bm25_score, 3),
                "Auth": "✓" if c.is_authoritative else "✗",
            }
            for c in trace.retrieved_candidates
        ]
        df = pd.DataFrame(records)
        st.sidebar.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.sidebar.caption("No KB retrieval in this turn.")

    # Tool calls
    st.sidebar.markdown("#### Tool Calls")
    if trace.tool_calls:
        for idx, tc in enumerate(trace.tool_calls, 1):
            st.sidebar.markdown(f"**{idx}. `{tc.name}`**")
            st.sidebar.json(tc.args)
        if trace.sanitized_tool_results:
            st.sidebar.markdown("**Sanitized Tool Result:**")
            st.sidebar.json(trace.sanitized_tool_results)
    else:
        st.sidebar.caption("No tool calls.")

    # Validation
    st.sidebar.markdown("#### Safety & Validation")
    if trace.validation_failures:
        for failure in trace.validation_failures:
            st.sidebar.error(failure)
    else:
        st.sidebar.success("Passed validation without flags.")


def main() -> None:
    st.set_page_config(
        page_title="Aster & Row Support Agent",
        page_icon="👜",
        layout="wide",
    )

    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"web_{uuid.uuid4().hex[:8]}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "latest_trace" not in st.session_state:
        st.session_state.latest_trace = None

    # Sidebar controls
    with st.sidebar:
        st.title("👜 Aster & Row")
        st.caption(f"**Session ID:** `{st.session_state.session_id}`")

        if st.button("🔄 Start New Session", use_container_width=True):
            st.session_state.session_id = f"web_{uuid.uuid4().hex[:8]}"
            st.session_state.messages = []
            st.session_state.latest_trace = None
            st.rerun()

        st.divider()
        show_debug = st.toggle("Show debug trace", value=False)

        if show_debug and st.session_state.latest_trace is not None:
            _render_debug_sidebar(st.session_state.latest_trace)
        elif show_debug:
            st.info("No turns executed yet. Send a message to see the debug trace.")

    # Main Chat View
    st.title("Aster & Row Customer Support")
    st.caption("Ask questions about bags, drinkware, travel accessories, policies, or order status.")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.markdown("**Sources:**")
                for cit in msg["citations"]:
                    st.markdown(f"- `[{cit}]`")
            if msg.get("human_handoff"):
                st.warning(f"⚠️ **Human Assistance Recommended:** {HUMAN_HANDOFF_MESSAGE}")

    # Chat input
    if prompt := st.chat_input("How can we help you today?"):
        # Display user message immediately
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call orchestrator
        with st.chat_message("assistant"):
            with st.spinner("Finding answer..."):
                response, trace = handle_message_with_trace(
                    session_id=st.session_state.session_id,
                    message=prompt,
                )

            st.markdown(response.text)
            if response.citations:
                st.markdown("**Sources:**")
                for cit in response.citations:
                    st.markdown(f"- `[{cit}]`")
            if response.human_handoff:
                st.warning(f"⚠️ **Human Assistance Recommended:** {HUMAN_HANDOFF_MESSAGE}")

        # Record assistant response & trace
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.text,
            "citations": response.citations,
            "human_handoff": response.human_handoff,
        })
        st.session_state.latest_trace = trace
        st.rerun()


if __name__ == "__main__":
    main()
