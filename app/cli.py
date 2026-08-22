"""
app/cli.py
----------
Interactive command-line interface for Aster & Row support agent.

Uses Typer for CLI arguments and Rich for terminal formatting.
Contains NO business logic — purely wraps handle_message_with_trace().
"""

from __future__ import annotations

import sys
from pathlib import Path
import uuid
from typing import Optional

# Ensure project root is in sys.path when launched directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.agent.orchestrator import handle_message_with_trace
from app.observability.trace import Trace

app = typer.Typer(
    help="Aster & Row RAG Support Agent CLI",
    add_completion=False,
)
console = Console()

HUMAN_HANDOFF_MESSAGE = (
    "To give you the most accurate help with this, "
    "I’m going to get a human expert on the line for you."
)


def _print_debug_trace(trace: Trace) -> None:
    """Pretty-print execution trace details using Rich tables and panels."""
    console.print("\n[bold yellow]─── Debug Trace ──────────────────────────────────────────[/bold yellow]")
    console.print(f"[bold cyan]Route Decision:[/bold cyan] {trace.route_decision}")
    if trace.handoff_reason:
        console.print(f"[bold cyan]Handoff Reason:[/bold cyan] {trace.handoff_reason}")

    # Retrieved candidates table
    if trace.retrieved_candidates:
        table = Table(
            title="Retrieved Knowledge Base Candidates",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Document / Heading", style="dim")
        table.add_column("Final", justify="right")
        table.add_column("Dense", justify="right")
        table.add_column("BM25", justify="right")
        table.add_column("Auth", justify="center")

        for c in trace.retrieved_candidates:
            doc_ref = f"{c.filename}#{c.heading}" if c.heading else c.filename
            auth_marker = "✓" if c.is_authoritative else "✗"
            table.add_row(
                doc_ref,
                f"{c.final_score:.3f}",
                f"{c.dense_score:.3f}",
                f"{c.bm25_score:.3f}",
                auth_marker,
            )
        console.print(table)
    else:
        console.print("[dim]Retrieved Candidates: None[/dim]")

    # Tool calls & sanitized results
    if trace.tool_calls:
        console.print("\n[bold cyan]Tool Calls:[/bold cyan]")
        for tc in trace.tool_calls:
            console.print(f"  • [bold]{tc.name}[/bold] args={tc.args}")
        if trace.sanitized_tool_results:
            console.print(f"  [bold green]Sanitized Result:[/bold green] {trace.sanitized_tool_results}")
    else:
        console.print("[dim]Tool Calls: None[/dim]")

    # Validation
    if trace.validation_failures:
        console.print("\n[bold red]Validation Failures / Flags:[/bold red]")
        for flag in trace.validation_failures:
            console.print(f"  • {flag}")
    else:
        console.print("[bold green]Validation: Passed cleanly[/bold green]")

    console.print("[bold yellow]──────────────────────────────────────────────────────────[/bold yellow]\n")


@app.command()
def main(
    session: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help="Session ID to isolate or resume a conversation.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Display pipeline debug trace for each turn.",
    ),
) -> None:
    """Run an interactive chat session with the Aster & Row support agent."""
    session_id = session or f"cli_{uuid.uuid4().hex[:8]}"

    console.print(
        Panel.fit(
            f"[bold green]Aster & Row Support Agent[/bold green]\n"
            f"[dim]Session ID:[/dim] [cyan]{session_id}[/cyan]\n"
            f"[dim]Debug Mode:[/dim] [cyan]{'Enabled' if debug else 'Disabled'}[/cyan]\n"
            f"[dim]Type 'exit' or 'quit' (or press Ctrl+C) to end session.[/dim]",
            title="Welcome",
            border_style="green",
        )
    )

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Ending session. Goodbye![/dim]")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Ending session. Goodbye![/dim]")
            sys.exit(0)

        try:
            response, trace = handle_message_with_trace(
                session_id=session_id,
                message=user_input,
            )
        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            continue

        console.print(f"\n[bold green]Agent:[/bold green] {response.text}")

        # Render citations
        if response.citations:
            console.print("\n[bold]Sources:[/bold]")
            for citation in response.citations:
                console.print(f"  [{citation}]")

        # Render human handoff message if escalation triggered
        if response.human_handoff:
            console.print(f"\n[bold yellow]{HUMAN_HANDOFF_MESSAGE}[/bold yellow]")

        # Render debug trace if enabled
        if debug:
            _print_debug_trace(trace)

        console.print()


if __name__ == "__main__":
    app()
