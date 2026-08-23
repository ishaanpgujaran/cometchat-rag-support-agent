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


def _render_debug_sidebar(trace: Trace) -> None:
    """Render the execution trace for the last turn in the sidebar."""
    route = trace.route_decision or "UNKNOWN"
    route_colors = {
        "KNOWLEDGE_LOOKUP": "green",
        "ORDER_LOOKUP": "blue",
        "NEEDS_ORDER_ID": "orange",
        "UNSAFE_OR_UNSUPPORTED": "red",
        "ABSTAIN_NO_EVIDENCE": "red",
    }
    badge_color = route_colors.get(route, "gray")
    st.sidebar.markdown(f"**Route Decision:** :{badge_color}[**{route}**]")

    if trace.handoff_reason:
        st.sidebar.caption(f"**Handoff Reason:** {trace.handoff_reason}")

    # Retrieved candidates
    st.sidebar.markdown("#### Retrieved Candidates")
    if trace.retrieved_candidates:
        records = [
            {
                "Filename": c.filename if not c.heading else f"{c.filename}#{c.heading}",
                "Final Score": round(c.final_score, 3),
                "Auth": "✓" if c.is_authoritative else "✗",
            }
            for c in trace.retrieved_candidates
        ]
        df = pd.DataFrame(records)
        st.sidebar.dataframe(df, use_container_width=True, hide_index=True, height=130)
    else:
        st.sidebar.caption("No KB retrieval in this turn.")

    # Tool calls
    st.sidebar.markdown("#### Tool Calls")
    if trace.tool_calls:
        for idx, tc in enumerate(trace.tool_calls, 1):
            order_id = tc.args.get("order_id", "")
            if order_id:
                st.sidebar.markdown(f"{idx}. `{tc.name}` (order_id: `{order_id}`)")
            else:
                args_str = ", ".join(f"{k}={v}" for k, v in tc.args.items()) if tc.args else ""
                st.sidebar.markdown(f"{idx}. `{tc.name}`" + (f" ({args_str})" if args_str else ""))
        if trace.sanitized_tool_results:
            st.sidebar.markdown("**Sanitized Tool Result:**")
            st.sidebar.json(trace.sanitized_tool_results)
    else:
        st.sidebar.caption("No tool calls in this turn.")

    # Validation
    st.sidebar.markdown("#### Validation")
    if trace.validation_failures:
        for failure in trace.validation_failures:
            st.sidebar.error(failure)
    else:
        st.sidebar.success("Validation passed cleanly")


def main() -> None:
    st.set_page_config(
        page_title="Aster & Row Support",
        page_icon="🏔️",
        layout="centered",
    )

    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"web_{uuid.uuid4().hex[:8]}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "latest_trace" not in st.session_state:
        st.session_state.latest_trace = None

    # Sidebar: Debug Panel & Session Controls
    with st.sidebar:
        st.header("Debug Trace")
        debug_on = st.toggle("Show trace for last turn", value=False)
        if debug_on:
            if st.session_state.latest_trace is not None:
                _render_debug_sidebar(st.session_state.latest_trace)
            else:
                st.info("No turns executed yet. Send a message to see the debug trace.")

        st.divider()
        if st.button("🔄 New Conversation", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Main Chat View Header
    st.title("🏔️ Aster & Row")
    st.caption("AI Customer Support · Powered by RAG")
    st.divider()

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            if msg.get("human_handoff"):
                st.warning("⚡ A human support specialist should assist with this request.")
            with st.chat_message("assistant", avatar="🏔️"):
                st.markdown(msg["content"])
                citations = msg.get("citations", [])
                if citations:
                    with st.expander(f"📎 {len(citations)} source(s) cited", expanded=False):
                        for c in citations:
                            st.markdown(f"- `{c}`")
        else:
            with st.chat_message("user"):
                st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about orders, returns, shipping..."):
        # Record user turn immediately
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.spinner("Thinking..."):
            try:
                response, trace = handle_message_with_trace(
                    session_id=st.session_state.session_id,
                    message=prompt,
                )
                st.session_state.latest_trace = trace
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.text,
                    "citations": response.citations,
                    "human_handoff": response.human_handoff,
                })
            except Exception as exc:
                err_msg = (
                    "⚠️ An error occurred while processing your request. "
                    f"Please try again in a moment. (Error: {exc})"
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                    "citations": [],
                    "human_handoff": True,
                })
        st.rerun()


if __name__ == "__main__":
    main()
