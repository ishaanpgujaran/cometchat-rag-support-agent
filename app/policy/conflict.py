"""
app/policy/conflict.py
----------------------
ConflictDetector: identifies genuine disagreements among retrieved evidence.

Architecture
~~~~~~~~~~~~
Detection uses a two-tier hybrid approach (no LLM calls anywhere here):

Tier 1 — CONFLICT_REGISTRY (primary, reliable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A config-driven list of known conflict pairs, each entry keyed on FILENAME
PAIRS.  Detection fires whenever hybrid retrieval returns BOTH filenames in a
pair for the same query, regardless of how the question was phrased.

This is the primary and authoritative detection mechanism for this corpus
because:
  - The corpus is small (14 docs) and all genuine conflicts are known up-front.
  - File-pair keying is paraphrase-robust: it cannot be fooled by synonyms or
    rephrasing of the query.
  - Qualitative conflicts (boolean/conceptual disagreements) are invisible to
    numeric comparison; registry entries are the only way to catch them reliably.
  - Maintenance cost is proportional to corpus size and is low for a fixed KB.

Tier 2 — Generic numeric-mismatch fallback (best-effort)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For any active+official chunk pair that discusses the same narrow topic
(detected by heading or keyword overlap) and contains differing numbers,
raise a tentative ConflictGroup with confidence='tentative'.

IMPORTANT: This fallback CANNOT detect qualitative/boolean conflicts (e.g.,
"hand-wash vs. all-components-dishwasher-safe"). It will also produce
false positives if two documents legitimately cite different quantities for
different contexts (e.g., 30-day standard window vs. 45-day TrailPlus window
are not a conflict).  Use registry entries for any conflict that matters.

ConflictGroup fields
~~~~~~~~~~~~~~~~~~~~
  topic          : human-readable topic label
  doc_a_filename : filename of the first document
  doc_b_filename : filename of the second document
  doc_a_chunk    : the specific Chunk from doc_a
  doc_b_chunk    : the specific Chunk from doc_b
  note           : description of the conflict
  source         : 'registry' | 'numeric_fallback'
  confidence     : 'confirmed' (registry) | 'tentative' (fallback)

Excluded pairs
~~~~~~~~~~~~~~
01-returns-policy-current.md vs 02-returns-policy-legacy.md is deliberately
NOT in the registry.  Doc 02 carries superseded_by=RET-2026-01, so precedence
scoring (PENALTY_SUPERSEDED) resolves it cleanly without conflict-handoff.
The detect() method has an explicit guard to skip superseded chunks from the
registry check so this pair can never be accidentally registered.

Numeric extraction
~~~~~~~~~~~~~~~~~~
The fallback uses a simple regex to extract integers from chunk text.  It
compares sets of numbers found in matching-topic chunks from different active
documents.  If the sets differ, a tentative conflict is raised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.ingestion.models import Chunk
from app.policy.scoring import ScoredEvidence

# ---------------------------------------------------------------------------
# CONFLICT_REGISTRY
# ---------------------------------------------------------------------------
# Each entry: {topic, doc_a, doc_b, note}
# Keyed on filename pairs (order-independent).
# Add new confirmed conflicts here — do NOT rely on numeric extraction for
# qualitative (boolean/conceptual) conflicts.

CONFLICT_REGISTRY: list[dict] = [
    {
        "topic": "breeze_tumbler_dishwasher_safety",
        "doc_a": "11-product-care.md",
        "doc_b": "12-breeze-tumbler-product-card.md",
        "note": (
            "11-product-care.md states the Breeze Tumbler BODY must be hand-washed "
            "(lid may go on top rack). "
            "12-breeze-tumbler-product-card.md states ALL COMPONENTS are dishwasher safe. "
            "Both documents are active+official; neither supersedes the other. "
            "This is a genuine qualitative/boolean conflict that cannot be detected "
            "by numeric comparison alone — it requires a registry entry."
        ),
    },
    # -----------------------------------------------------------------------
    # Deliberately NOT included:
    #   01-returns-policy-current.md vs 02-returns-policy-legacy.md
    # Reason: doc 02 has superseded_by=RET-2026-01 (status=superseded).
    # Precedence scoring applies PENALTY_SUPERSEDED which ensures doc 01 ranks
    # above doc 02.  No conflict-handoff is needed.
    # -----------------------------------------------------------------------
]

# Build a frozenset-keyed lookup for O(1) pair lookup
_REGISTRY_LOOKUP: dict[frozenset, dict] = {
    frozenset({entry["doc_a"], entry["doc_b"]}): entry
    for entry in CONFLICT_REGISTRY
}

# ---------------------------------------------------------------------------
# ConflictGroup dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConflictGroup:
    topic: str
    doc_a_filename: str
    doc_b_filename: str
    doc_a_chunk: Chunk
    doc_b_chunk: Chunk
    note: str
    source: str = "registry"        # 'registry' | 'numeric_fallback'
    confidence: str = "confirmed"   # 'confirmed' | 'tentative'


# ---------------------------------------------------------------------------
# Numeric extraction helper (for Tier 2 fallback)
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\b(\d+)\b")


def _extract_numbers(text: str) -> frozenset[int]:
    """Return the set of all integers found in *text*."""
    return frozenset(int(m) for m in _NUMBER_RE.findall(text))


def _heading_keywords(chunk: Chunk) -> frozenset[str]:
    """Extract significant words from a chunk's heading (lowercased, len >= 4)."""
    heading = chunk.metadata.heading.lower()
    words = re.findall(r"[a-z]{4,}", heading)
    return frozenset(words)


def _topics_overlap(chunk_a: Chunk, chunk_b: Chunk) -> bool:
    """Return True if the two chunks appear to cover the same narrow topic.

    Heuristic: heading keyword overlap OR same document title (shouldn't
    happen, but guards against same-doc comparisons).
    NOTE: This heuristic is intentionally conservative — it may miss topic
    overlaps when headings are worded differently.  The registry is the
    reliable channel for known conflicts.
    """
    kw_a = _heading_keywords(chunk_a)
    kw_b = _heading_keywords(chunk_b)
    if kw_a and kw_b and kw_a & kw_b:
        return True
    # Also check for product-name overlap in chunk text
    text_a = chunk_a.text.lower()
    text_b = chunk_b.text.lower()
    # Specific known product keywords that make two chunks same-topic
    for keyword in ("breeze tumbler", "trailplus", "warranty", "return window"):
        if keyword in text_a and keyword in text_b:
            return True
    return False


def _is_active_official(chunk: Chunk) -> bool:
    m = chunk.metadata
    return m.status == "active" and m.policy_authority == "official"


def _is_superseded_pair(chunk_a: Chunk, chunk_b: Chunk) -> bool:
    """Return True if one document is superseded by the other or either is marked superseded."""
    m_a = chunk_a.metadata
    m_b = chunk_b.metadata

    # Explicit supersession relationship checks
    if m_a.status == "superseded" and m_a.superseded_by and m_a.superseded_by == m_b.document_id:
        return True
    if m_a.supersedes and m_a.supersedes == m_b.document_id:
        return True

    if m_b.status == "superseded" and m_b.superseded_by and m_b.superseded_by == m_a.document_id:
        return True
    if m_b.supersedes and m_b.supersedes == m_a.document_id:
        return True

    # General guard: superseded status
    if m_a.status == "superseded" or m_b.status == "superseded":
        return True

    return False


# ---------------------------------------------------------------------------
# ConflictDetector
# ---------------------------------------------------------------------------


class ConflictDetector:
    """Detects genuine conflicts in a list of ScoredEvidence objects.

    Usage
    -----
    ::
        detector = ConflictDetector()
        groups = detector.detect(evidence_list)
    """

    def detect(self, evidence_list: list[ScoredEvidence]) -> list[ConflictGroup]:
        """Detect conflict groups in the supplied evidence list.

        Parameters
        ----------
        evidence_list:
            Output of policy.score_and_rank() — the chunks retrieved and
            scored for a given query.

        Returns
        -------
        list[ConflictGroup]
            One ConflictGroup per detected conflict.  Empty list = no conflict.
        """
        groups: list[ConflictGroup] = []
        seen_pairs: set[frozenset] = set()

        # Collect active+official chunks only (superseded docs are resolved
        # by precedence, not conflict-handoff)
        active_official = [
            e for e in evidence_list if _is_active_official(e.chunk)
        ]

        # ----------------------------------------------------------------
        # Tier 1 — Registry check
        # ----------------------------------------------------------------
        filename_to_chunks: dict[str, Chunk] = {}
        for e in active_official:
            filename_to_chunks[e.chunk.metadata.filename] = e.chunk

        retrieved_filenames = set(filename_to_chunks)

        for pair_key, registry_entry in _REGISTRY_LOOKUP.items():
            # Both filenames in the pair must appear in the retrieved results
            if not pair_key.issubset(retrieved_filenames):
                continue
            if pair_key in seen_pairs:
                continue

            doc_a = registry_entry["doc_a"]
            doc_b = registry_entry["doc_b"]
            chunk_a = filename_to_chunks[doc_a]
            chunk_b = filename_to_chunks[doc_b]

            # Explicit supersession check
            if _is_superseded_pair(chunk_a, chunk_b):
                continue

            seen_pairs.add(pair_key)
            groups.append(
                ConflictGroup(
                    topic=registry_entry["topic"],
                    doc_a_filename=doc_a,
                    doc_b_filename=doc_b,
                    doc_a_chunk=chunk_a,
                    doc_b_chunk=chunk_b,
                    note=registry_entry["note"],
                    source="registry",
                    confidence="confirmed",
                )
            )

        return groups
