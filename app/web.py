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

_CUSTOM_CSS = """
<style>
/* Modern typography & font smoothing */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Subtle header styling */
.header-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.25rem;
}

.brand-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(34, 197, 94, 0.1);
    color: #16a34a;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 9999px;
    border: 1px solid rgba(34, 197, 94, 0.2);
}

.status-dot {
    width: 6px;
    height: 6px;
    background-color: #22c55e;
    border-radius: 50%;
    display: inline-block;
}

/* Subtle card styling for welcome prompts */
.suggestion-box {
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 1.2rem;
    background: rgba(248, 250, 252, 0.5);
    margin-bottom: 1rem;
}

@media (prefers-color-scheme: dark) {
    .suggestion-box {
        background: rgba(30, 41, 59, 0.3);
        border-color: rgba(51, 65, 85, 0.5);
    }
}

/* Refined chat message styling */
[data-testid="stChatMessage"] {
    border-radius: 10px;
    padding: 0.85rem 1rem;
}

/* Sidebar polish */
[data-testid="stSidebar"] {
    padding-top: 1rem;
}
</style>
"""


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

    # Inject custom minimal styling
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

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
        st.caption(f"**Session:** `{st.session_state.session_id}`")
        if st.button("🔄 New Conversation", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Main Chat View Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏔️ Aster & Row")
        st.caption("AI Customer Support · Powered by RAG")
    with col2:
        st.markdown(
            '<div style="text-align: right; padding-top: 1.5rem;">'
            '<span class="brand-status"><span class="status-dot"></span> Live Assistant</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Starter prompts for an empty conversation
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="suggestion-box">
                <p style="margin: 0 0 0.5rem 0; font-size: 0.9rem; font-weight: 600;">👋 How can we help today?</p>
                <p style="margin: 0; font-size: 0.82rem; color: #64748b;">
                    Ask questions about order status, return policies, warranty coverage, or shipping information.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📦 Where is my order ORD-1001?", use_container_width=True):
                st.session_state.pending_prompt = "Where is my order ORD-1001?"
                st.rerun()
            if st.button("🔄 International return policy", use_container_width=True):
                st.session_state.pending_prompt = "What is the return policy for international orders?"
                st.rerun()
        with col_b:
            if st.button("🛡️ Warranty on broken zipper", use_container_width=True):
                st.session_state.pending_prompt = "Is a broken zipper covered under the warranty?"
                st.rerun()
            if st.button("⚡ Cancel my order ORD-1002", use_container_width=True):
                st.session_state.pending_prompt = "I would like to cancel order ORD-1002"
                st.rerun()

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

    # Handle pending prompt from starter chips or text input
    pending = st.session_state.pop("pending_prompt", None)
    prompt = st.chat_input("Ask about orders, returns, shipping...") or pending

    if prompt:
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
