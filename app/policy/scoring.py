"""
app/policy/scoring.py
---------------------
Precedence scoring and evidence ranking for retrieved knowledge-base chunks.

Scoring formula
~~~~~~~~~~~~~~~
    final_score = ALPHA * dense_score + BETA * bm25_score
                  + metadata_bonus - metadata_penalty

Named constants (tune here):

    ALPHA  — weight for dense (semantic) similarity component
    BETA   — weight for BM25 (lexical) similarity component
    ALPHA + BETA ≈ 1.0 recommended; do not need to sum exactly to 1.

Bonus / penalty values (additive on top of retrieval score):

    BONUS_ACTIVE_OFFICIAL     — chunk is status=active AND policy_authority=official
    BONUS_CUSTOMER_FACING     — chunk has customer_answering=True
    PENALTY_SUPERSEDED        — chunk is status=superseded
    PENALTY_INTERNAL          — chunk has audience=internal
    PENALTY_DRAFT             — chunk is status=draft
    PENALTY_NOT_OFFICIAL      — chunk is policy_authority != 'official'

These are additive so multiple bonuses/penalties can stack.  Scores can be
negative (heavily penalised documents will correctly rank below zero).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.ingestion.models import Chunk
from app.retrieval.models import RetrievedCandidate

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

ALPHA: float = 0.65  # dense / semantic weight
BETA: float = 0.35   # BM25 / lexical weight

# Bonuses (positive)
BONUS_ACTIVE_OFFICIAL: float = 0.20   # active + official → highest reliability
BONUS_CUSTOMER_FACING: float = 0.05   # customer_answering=True → safe to surface

# Penalties (positive magnitude; subtracted in formula)
PENALTY_SUPERSEDED: float = 0.40      # superseded docs must never win over active ones
PENALTY_INTERNAL: float = 0.30        # internal docs must not surface to customers
PENALTY_DRAFT: float = 0.35           # draft docs are unreviewed and untrustworthy
PENALTY_NOT_OFFICIAL: float = 0.25    # non-official authority reduces credibility


# ---------------------------------------------------------------------------
# ScoredEvidence model
# ---------------------------------------------------------------------------

class ScoredEvidence(BaseModel):
    """A retrieval candidate annotated with its final policy-aware score."""

    chunk: Chunk
    bm25_score: float
    dense_score: float
    metadata_bonus: float = Field(description="Total bonus applied")
    metadata_penalty: float = Field(description="Total penalty applied")
    final_score: float = Field(
        description=(
            "ALPHA * dense_score + BETA * bm25_score + metadata_bonus - metadata_penalty"
        )
    )

    @property
    def is_customer_safe(self) -> bool:
        """True when this evidence may be surfaced in a customer-facing response."""
        m = self.chunk.metadata
        return (
            m.customer_answering
            and m.audience == "customer"
            and m.status not in ("superseded", "draft")
            and m.policy_authority == "official"
        )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _compute_bonus(chunk: Chunk) -> float:
    m = chunk.metadata
    bonus = 0.0
    if m.status == "active" and m.policy_authority == "official":
        bonus += BONUS_ACTIVE_OFFICIAL
    if m.customer_answering:
        bonus += BONUS_CUSTOMER_FACING
    return bonus


def _compute_penalty(chunk: Chunk) -> float:
    m = chunk.metadata
    penalty = 0.0
    if m.status == "superseded":
        penalty += PENALTY_SUPERSEDED
    if m.audience == "internal":
        penalty += PENALTY_INTERNAL
    if m.status == "draft":
        penalty += PENALTY_DRAFT
    if m.policy_authority != "official":
        penalty += PENALTY_NOT_OFFICIAL
    return penalty


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_candidate(candidate: RetrievedCandidate) -> ScoredEvidence:
    """Apply the scoring formula to a single RetrievedCandidate."""
    bonus = _compute_bonus(candidate.chunk)
    penalty = _compute_penalty(candidate.chunk)
    final = (
        ALPHA * candidate.dense_score
        + BETA * candidate.bm25_score
        + bonus
        - penalty
    )
    return ScoredEvidence(
        chunk=candidate.chunk,
        bm25_score=candidate.bm25_score,
        dense_score=candidate.dense_score,
        metadata_bonus=bonus,
        metadata_penalty=penalty,
        final_score=final,
    )


def score_and_rank(candidates: list[RetrievedCandidate]) -> list[ScoredEvidence]:
    """Score and rank a list of RetrievedCandidates.

    Returns them sorted by final_score descending (highest ranked first).
    """
    scored = [score_candidate(c) for c in candidates]
    scored.sort(key=lambda e: e.final_score, reverse=True)
    return scored


def is_authoritative(evidence: ScoredEvidence) -> bool:
    """Return True if this evidence is suitable as a primary citation source.

    Criteria (all must hold):
    - status = 'active'
    - policy_authority = 'official'
    - audience = 'customer'
    - customer_answering = True
    - final_score > 0
    """
    m = evidence.chunk.metadata
    return (
        m.status == "active"
        and m.policy_authority == "official"
        and m.audience == "customer"
        and m.customer_answering
        and evidence.final_score > 0
    )
