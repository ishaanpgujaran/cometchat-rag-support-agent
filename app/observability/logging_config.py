"""
app/observability/logging_config.py
------------------------------------
Structured JSON logging for the support-agent pipeline.

Design
~~~~~~
Uses Python's stdlib ``logging`` module with a custom ``JsonFormatter`` so that
every log record emitted via ``get_json_logger()`` produces exactly ONE
JSON-serialisable line.

One log line is emitted per pipeline stage by the orchestrator.  Each line
carries:
  - Standard fields: timestamp, level, logger, message (human stage label)
  - Pipeline fields: trace_id, session_id, stage
  - Stage-specific payload (scores, counts, flag names, citation refs, etc.)

Privacy rules (hard-coded in the formatter and enforced at call sites)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NEVER log:
  * The raw GEMINI_API_KEY value.
  * Any raw order field not present in SafeOrderResult (risk_score,
    internal_notes, warehouse_note, support_tags, customer email/address).
  * The full text of any knowledge-base document chunk.
    Callers MUST pass ``filename#heading`` references instead of chunk text.

Usage
~~~~~
    from app.observability.logging_config import configure_json_logging, get_json_logger

    # Call once at process startup (idempotent):
    configure_json_logging()

    logger = get_json_logger(__name__)
    logger.info("stage=routed", extra={"trace_id": t, "session_id": s,
                                        "stage": "route",
                                        "route_decision": "KNOWLEDGE_LOOKUP"})
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """
    Formats every LogRecord as a single-line JSON object.

    Standard fields always present:
        ts          — ISO-8601 UTC timestamp
        level       — log level name (INFO, WARNING, ERROR, …)
        logger      — logger name (module path)
        message     — the human-readable log message

    Extra fields are taken from ``record.__dict__`` for any key listed in
    ``PIPELINE_EXTRA_KEYS`` below.  All other keys from ``extra={}`` are
    also forwarded so callers can attach arbitrary stage-specific data.

    The formatter explicitly EXCLUDES a deny-list of keys that could
    accidentally carry sensitive data.
    """

    # Keys that are always excluded from the JSON output regardless of source.
    # This is a last-resort safety net; primary enforcement is at call sites.
    _DENIED_KEYS: frozenset[str] = frozenset({
        "GEMINI_API_KEY",
        "api_key",
        "apikey",
        "risk_score",
        "riskscore",
        "internal_notes",
        "internalnotes",
        "warehouse_note",
        "warehousenote",
        "support_tags",
        "supporttags",
        "password",
        "secret",
        "token",
        # stdlib LogRecord internals that are never useful in JSON
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    })

    def format(self, record: logging.LogRecord) -> str:
        record.getMessage()  # interpolate record.msg % record.args if any
        doc: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach exception info when present
        if record.exc_info:
            doc["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            doc["exception"] = record.exc_text

        # Forward all extra fields not in the deny-list
        for key, value in record.__dict__.items():
            if key in self._DENIED_KEYS:
                continue
            if key.startswith("_"):
                continue
            if key in doc:
                continue
            # Only scalars / simple types to keep JSON clean
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                doc[key] = value

        return json.dumps(doc, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

_CONFIGURED: bool = False


def configure_json_logging(
    level: int = logging.INFO,
    handler: logging.Handler | None = None,
) -> None:
    """
    Install the JSON formatter on the root logger.

    This function is idempotent — calling it multiple times is safe.
    In tests you can pass a custom ``handler`` (e.g. a ``logging.handlers.MemoryHandler``
    or a simple list-appending handler) to capture structured log output.

    Parameters
    ----------
    level:
        Root logging level.  Defaults to ``logging.INFO``.
    handler:
        Optional pre-built handler.  When None, a ``StreamHandler`` writing
        to stderr is created.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    fmt = JsonFormatter()

    if handler is None:
        handler = logging.StreamHandler()

    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicating handlers in test environments that call this repeatedly
    root.addHandler(handler)

    _CONFIGURED = True


def get_json_logger(name: str) -> logging.Logger:
    """
    Return a logger for *name* that emits structured JSON lines.

    The returned logger inherits from the root logger configured by
    ``configure_json_logging()``.  If ``configure_json_logging()`` has not
    been called yet, log output will use the root logger's existing handlers
    (which may not be JSON-formatted); call ``configure_json_logging()`` at
    process startup to ensure consistent formatting.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    """
    return logging.getLogger(name)
