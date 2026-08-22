"""
tests/unit/test_interfaces.py
-----------------------------
Unit tests for CLI (app/cli.py) and Streamlit Web UI (app/web.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner

from app.agent.orchestrator import AgentResponse
from app.cli import app, HUMAN_HANDOFF_MESSAGE
from app.observability.trace import CandidateRef, Trace, ToolCallRef
from app.web import _render_debug_sidebar


runner = CliRunner()


def _make_sample_trace(session_id: str = "test-session") -> Trace:
    trace = Trace(
        session_id=session_id,
        user_message="Where is ORD-1007?",
        route_decision="ORDER_LOOKUP",
    )
    trace.retrieved_candidates = [
        CandidateRef(
            filename="05-domestic-shipping.md",
            heading="Standard Shipping",
            document_id="DOC-05",
            dense_score=0.82,
            bm25_score=0.74,
            final_score=0.85,
            is_authoritative=True,
            audience="customer",
            status="active",
        )
    ]
    trace.tool_calls = [
        ToolCallRef(name="lookup_order", args={"order_id": "ORD-1007"})
    ]
    trace.sanitized_tool_results = {
        "order_id": "ORD-1007",
        "status": "shipped",
        "carrier": "UPS",
    }
    trace.validation_failures = []
    return trace


def test_cli_exit_command():
    """CLI exits cleanly when user inputs exit."""
    result = runner.invoke(app, input="exit\n")
    assert result.exit_code == 0
    assert "Ending session. Goodbye!" in result.output


def test_cli_quit_command():
    """CLI exits cleanly when user inputs quit."""
    result = runner.invoke(app, input="quit\n")
    assert result.exit_code == 0
    assert "Ending session. Goodbye!" in result.output


def test_cli_custom_session():
    """CLI respects custom --session argument."""
    result = runner.invoke(app, ["--session", "custom-session-123"], input="exit\n")
    assert result.exit_code == 0
    assert "custom-session-123" in result.output


@patch("app.cli.handle_message_with_trace")
def test_cli_knowledge_question_with_citations(mock_handle):
    """CLI renders agent text and citations in [filename#heading] style."""
    sample_response = AgentResponse(
        text="Returns are accepted within 30 days of purchase.",
        citations=["01-returns-policy-current.md#Return Window"],
        human_handoff=False,
    )
    sample_trace = Trace(
        session_id="test-session",
        user_message="What is the return window?",
        route_decision="KNOWLEDGE_LOOKUP",
    )
    mock_handle.return_value = (sample_response, sample_trace)

    result = runner.invoke(app, input="What is the return window?\nexit\n")
    assert result.exit_code == 0
    assert "Returns are accepted within 30 days of purchase." in result.output
    assert "[01-returns-policy-current.md#Return Window]" in result.output
    normalized_output = " ".join(result.output.split())
    assert "To give you the most accurate help with this" not in normalized_output


@patch("app.cli.handle_message_with_trace")
def test_cli_order_lookup(mock_handle):
    """CLI renders order lookup answer without citations or false handoff."""
    sample_response = AgentResponse(
        text="Your order ORD-1007 has shipped via UPS and will arrive by Aug 30.",
        citations=[],
        human_handoff=False,
    )
    sample_trace = _make_sample_trace()
    mock_handle.return_value = (sample_response, sample_trace)

    result = runner.invoke(app, input="Where is ORD-1007?\nexit\n")
    assert result.exit_code == 0
    assert "Your order ORD-1007 has shipped" in result.output


@patch("app.cli.handle_message_with_trace")
def test_cli_multi_turn_flow(mock_handle):
    """CLI handles multi-turn conversation across multiple inputs."""
    resp1 = AgentResponse(
        text="Yes, we ship to select international destinations.",
        citations=["06-international-shipping.md#Eligible Countries"],
        human_handoff=False,
    )
    trace1 = Trace(session_id="test-session", user_message="Do you ship internationally?", route_decision="KNOWLEDGE_LOOKUP")

    resp2 = AgentResponse(
        text="Yes, we ship to Canada via DHL Express.",
        citations=["06-international-shipping.md#Canada & Mexico"],
        human_handoff=False,
    )
    trace2 = Trace(session_id="test-session", user_message="What about Canada?", route_decision="KNOWLEDGE_LOOKUP")

    mock_handle.side_effect = [(resp1, trace1), (resp2, trace2)]

    result = runner.invoke(app, input="Do you ship internationally?\nWhat about Canada?\nexit\n")
    assert result.exit_code == 0
    assert "Yes, we ship to select international destinations." in result.output
    assert "Yes, we ship to Canada via DHL Express." in result.output
    assert "[06-international-shipping.md#Eligible Countries]" in result.output
    assert "[06-international-shipping.md#Canada & Mexico]" in result.output
    assert mock_handle.call_count == 2


@patch("app.cli.handle_message_with_trace")
def test_cli_renders_human_handoff(mock_handle):
    """CLI prints the handoff message when human_handoff is True."""
    sample_response = AgentResponse(
        text="I cannot cancel this order directly.",
        citations=[],
        human_handoff=True,
    )
    sample_trace = _make_sample_trace()
    mock_handle.return_value = (sample_response, sample_trace)

    result = runner.invoke(app, input="Cancel my order ORD-1007\nexit\n")
    assert result.exit_code == 0
    assert "I cannot cancel this order directly." in result.output
    normalized_output = " ".join(result.output.split())
    assert HUMAN_HANDOFF_MESSAGE in normalized_output


@patch("app.cli.handle_message_with_trace")
def test_cli_debug_mode_prints_trace(mock_handle):
    """CLI prints debug trace including route decision, candidate scores, and tools."""
    sample_response = AgentResponse(
        text="Order ORD-1007 is shipped.",
        citations=[],
        human_handoff=False,
    )
    sample_trace = _make_sample_trace()
    mock_handle.return_value = (sample_response, sample_trace)

    result = runner.invoke(app, ["--debug"], input="Where is ORD-1007?\nexit\n")
    assert result.exit_code == 0
    assert "ORDER_LOOKUP" in result.output
    assert "05-domestic-shipping.md#Standard Shipping" in result.output
    assert "lookup_order" in result.output
    assert "Sanitized Result" in result.output


def test_web_debug_sidebar_rendering():
    """Streamlit debug sidebar renders without errors when given a trace."""
    trace = _make_sample_trace()
    trace.validation_failures = ["Test safety flag"]

    with patch("streamlit.sidebar.markdown") as mock_md, \
         patch("streamlit.sidebar.dataframe") as mock_df, \
         patch("streamlit.sidebar.json") as mock_json, \
         patch("streamlit.sidebar.error") as mock_err, \
         patch("streamlit.sidebar.warning") as mock_warn:
        _render_debug_sidebar(trace)
        assert mock_md.called
        assert mock_df.called
        assert mock_json.called
        assert mock_err.called
