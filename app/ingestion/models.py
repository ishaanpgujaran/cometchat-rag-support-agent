"""
app/ingestion/models.py
-----------------------
Pydantic data models for the knowledge-base ingestion pipeline.

Field mapping notes (verified against CORPUS_FACTS.md and actual front-matter):
- document_id   : required; direct front-matter field
- title         : required; direct front-matter field
- status        : required; direct front-matter field ("active", "superseded", "draft")
- audience      : required; direct front-matter field ("customer", "internal")
- policy_authority : required; direct front-matter field ("official", "none")
- effective_date    : optional; ISO date string, present on all observed docs
- last_reviewed     : optional; ISO date string, present on most docs
- supersedes        : optional; only on doc 01 (RET-2026-01)
- superseded_by     : optional; only on doc 02 (RET-2024-01)
- superseded_date   : optional; only on doc 02 (RET-2024-01)
- customer_answering: optional; only explicitly on doc 14 (False);
                      defaults to True for all docs where the field is absent.
                      This is the only field inferred — inference rule: absent => True.

Inference rules are documented here because CORPUS_FACTS.md designates
'customer_answering' as the only field that must be inferred (default True
when absent), and all other Optional fields are genuinely absent from some
documents, not inferred.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata for a single knowledge-base chunk.

    All field names use the exact strings from the front-matter YAML
    as documented in docs/CORPUS_FACTS.md.
    """

    # --- Required fields (present in every document) ---
    filename: str = Field(description="Basename of the source .md file, e.g. '01-returns-policy-current.md'")
    document_id: str = Field(description="Unique document identifier from front matter, e.g. 'RET-2026-01'")
    title: str = Field(description="Human-readable document title from front matter")
    status: str = Field(description="Document status: 'active', 'superseded', or 'draft'")
    audience: str = Field(description="Intended audience: 'customer' or 'internal'")
    policy_authority: str = Field(description="Document authority level: 'official' or 'none'")

    # --- Section-level fields (populated per chunk) ---
    heading: str = Field(description="The heading text of this chunk's section, or '' for the intro block")

    # --- Optional front-matter fields ---
    effective_date: Optional[str] = Field(default=None, description="ISO date when this policy became effective")
    last_reviewed: Optional[str] = Field(default=None, description="ISO date of last review")

    # --- Conditional fields (only on specific documents per CORPUS_FACTS.md) ---
    supersedes: Optional[str] = Field(default=None, description="document_id this doc supersedes (doc 01 only)")
    superseded_by: Optional[str] = Field(default=None, description="document_id that supersedes this doc (doc 02 only)")
    superseded_date: Optional[str] = Field(default=None, description="ISO date this doc was superseded (doc 02 only)")

    # --- Inferred field ---
    customer_answering: bool = Field(
        default=True,
        description=(
            "Whether this document may be used to answer customer queries. "
            "Explicitly False only in doc 14 (MIG-TEST-04). "
            "INFERENCE RULE: absent from front matter → default True. "
            "Explicitly set False for any doc with audience='internal' or "
            "policy_authority='none' as an additional safety guard."
        ),
    )


class Chunk(BaseModel):
    """A single retrievable unit of knowledge-base content."""

    chunk_id: str = Field(
        description=(
            "Stable unique identifier formed as '<filename_stem>__<heading_slug>' "
            "where heading_slug is the heading lowercased with spaces replaced by '_'. "
            "For the intro block (before first heading), heading_slug is 'intro'."
        )
    )
    text: str = Field(description="Full text content of this chunk including its heading line")
    metadata: ChunkMetadata
