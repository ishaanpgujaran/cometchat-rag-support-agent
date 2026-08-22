"""app/policy — precedence scoring, evidence ranking, and conflict detection."""

from app.policy.scoring import (
    ScoredEvidence,
    score_and_rank,
    score_candidate,
    is_authoritative,
    ALPHA,
    BETA,
    BONUS_ACTIVE_OFFICIAL,
    BONUS_CUSTOMER_FACING,
    PENALTY_SUPERSEDED,
    PENALTY_INTERNAL,
    PENALTY_DRAFT,
    PENALTY_NOT_OFFICIAL,
)
from app.policy.conflict import (
    ConflictDetector,
    ConflictGroup,
    CONFLICT_REGISTRY,
)

__all__ = [
    "ScoredEvidence",
    "score_and_rank",
    "score_candidate",
    "is_authoritative",
    "ALPHA",
    "BETA",
    "BONUS_ACTIVE_OFFICIAL",
    "BONUS_CUSTOMER_FACING",
    "PENALTY_SUPERSEDED",
    "PENALTY_INTERNAL",
    "PENALTY_DRAFT",
    "PENALTY_NOT_OFFICIAL",
    "ConflictDetector",
    "ConflictGroup",
    "CONFLICT_REGISTRY",
]
