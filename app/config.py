"""
app/config.py
-------------
Central configuration module.  Reads all settings from the environment
(populated from .env by python-dotenv).

Design rules
~~~~~~~~~~~~
* Importing this module must never fail due to a missing GEMINI_API_KEY.
* The key is only required at the moment a live network call is about to be
  made.  Callers that need the key should call ``require_api_key()`` just
  before using the Gemini client.
* All other settings have sensible defaults so the app works for local
  embedding / order-lookup testing without any credentials.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root (two levels up from this file: app/ → /)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Gemini / LLM settings
# ---------------------------------------------------------------------------

# May be None at import time — only required when making a network call.
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE_DIR: Path = Path(
    os.getenv("KNOWLEDGE_BASE_DIR", str(_PROJECT_ROOT / "knowledge-base"))
)

ORDERS_FILE_PATH: Path = Path(
    os.getenv("ORDERS_FILE_PATH", str(_PROJECT_ROOT / "data" / "orders.json"))
)


# ---------------------------------------------------------------------------
# Guard helper
# ---------------------------------------------------------------------------

def require_api_key() -> str:
    """Return the Gemini API key or raise a clear error.

    Call this immediately before any code that opens a network connection to
    the Gemini API.  Never call it at module import time.

    Raises
    ------
    EnvironmentError
        When GEMINI_API_KEY has not been set in the environment or .env file.
    """
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.  "
            "Add it to your .env file or export it in your shell before "
            "making a request that requires the Gemini API."
        )
    return GEMINI_API_KEY
