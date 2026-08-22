"""
app/session/store.py
--------------------
In-memory session store for the support agent.

Design
~~~~~~
* Keyed by session_id (any opaque string).
* Each session holds turn history + a small routing-context snapshot.
* Thread-safety: a per-session threading.Lock is used so that Session A
  can NEVER read or mutate Session B's state, even under concurrent access.
  The global dict itself is guarded by a lightweight creation-lock so that
  two threads racing to create the *same* session_id produce exactly one
  Session object.

Public API
~~~~~~~~~~
  get_session(session_id)          -> Session  (creates if absent)
  append_turn(session_id, role, content)
  update_context(session_id, **kwargs)

Singleton
~~~~~~~~~
  A module-level ``_store = SessionStore()`` is the default instance.
  Use the module-level helpers (get_session / append_turn / update_context)
  to access it without instantiating your own store.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SessionTurn:
    """A single conversational turn (user or assistant)."""
    role: str           # "user" | "assistant" | "system"
    content: str
    timestamp: str      # ISO-8601 UTC string


@dataclass
class SessionContext:
    """
    Lightweight routing context persisted across turns.

    Fields
    ------
    last_order_id : str | None
        The most-recently discussed order ID (normalised, uppercase).
    last_topic : str | None
        A short label for the last knowledge topic (e.g. "international shipping").
    last_route : str | None
        The RouteDecision value from the last turn, as a plain string
        (avoids an import cycle with app.agent.router).
    """
    last_order_id: Optional[str] = None
    last_topic: Optional[str] = None
    last_route: Optional[str] = None


@dataclass
class Session:
    """
    Full session state for one customer conversation.

    Attributes
    ----------
    session_id : str
        Opaque unique identifier.
    turns : list[SessionTurn]
        Chronological conversation history (user + assistant turns).
    context : SessionContext
        Compact routing context updated after each turn.
    _lock : threading.Lock
        Private per-session lock — callers must NOT hold this directly.
    """
    session_id: str
    turns: list[SessionTurn] = field(default_factory=list)
    context: SessionContext = field(default_factory=SessionContext)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

class SessionStore:
    """
    In-memory dict-backed store for Session objects.

    Thread-safety contract
    ~~~~~~~~~~~~~~~~~~~~~~
    * ``_creation_lock`` serialises concurrent first-access for the *same*
      session_id.  Two threads racing on an unknown ID will produce exactly
      one Session.
    * Each Session carries its own ``_lock``.  All reads/writes to a
      session's fields (``turns``, ``context``) are performed while holding
      ``session._lock``.  This guarantees that Session A's lock is NEVER
      held while accessing Session B's data.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._creation_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Session access
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> Session:
        """Return the Session for *session_id*, creating it if absent.

        This is the only entry-point for session creation.  The returned
        Session object is *live* — callers must not cache it across calls
        if they expect to see mutations from other threads.
        """
        if session_id in self._sessions:
            return self._sessions[session_id]
        with self._creation_lock:
            # Double-check after acquiring the lock (TOCTOU guard)
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
    ) -> None:
        """Append a new turn to the session history.

        Parameters
        ----------
        session_id:
            Target session (created if absent).
        role:
            "user", "assistant", or "system".
        content:
            The raw text of the turn.
        timestamp:
            ISO-8601 UTC string.  Defaults to current UTC time.
        """
        session = self.get_session(session_id)
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        turn = SessionTurn(role=role, content=content, timestamp=ts)
        with session._lock:
            session.turns.append(turn)

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def update_context(
        self,
        session_id: str,
        last_order_id: Optional[str] = None,
        last_topic: Optional[str] = None,
        last_route: Optional[str] = None,
    ) -> None:
        """Update the routing context for a session.

        Only keyword arguments that are explicitly passed (and non-None)
        will overwrite the existing values.  Pass None explicitly to clear
        a field (e.g., ``last_order_id=None`` resets it).

        Note: to *clear* a field pass the sentinel ``""`` rather than None,
        because ``None`` is also the sentinel for "not provided".
        To unambiguously clear, use the ``_clear_context_field`` helper or
        pass the empty string.
        """
        session = self.get_session(session_id)
        with session._lock:
            if last_order_id is not None:
                session.context.last_order_id = last_order_id or None
            if last_topic is not None:
                session.context.last_topic = last_topic or None
            if last_route is not None:
                session.context.last_route = last_route or None

    def get_recent_turns(self, session_id: str, n: int = 4) -> list[SessionTurn]:
        """Return the last *n* turns (default 4) for prompt construction.

        Acquires the session lock for a consistent snapshot.
        """
        session = self.get_session(session_id)
        with session._lock:
            return list(session.turns[-n:])


# ---------------------------------------------------------------------------
# Module-level singleton + helper functions
# ---------------------------------------------------------------------------

_store = SessionStore()


def get_session(session_id: str) -> Session:
    """Return (or create) the Session for *session_id* from the default store."""
    return _store.get_session(session_id)


def append_turn(
    session_id: str,
    role: str,
    content: str,
    timestamp: Optional[str] = None,
) -> None:
    """Append a turn to the default store's session."""
    _store.append_turn(session_id, role, content, timestamp)


def update_context(
    session_id: str,
    last_order_id: Optional[str] = None,
    last_topic: Optional[str] = None,
    last_route: Optional[str] = None,
) -> None:
    """Update routing context in the default store."""
    _store.update_context(session_id, last_order_id, last_topic, last_route)


def get_recent_turns(session_id: str, n: int = 4) -> list[SessionTurn]:
    """Return the last *n* turns from the default store."""
    return _store.get_recent_turns(session_id, n)
