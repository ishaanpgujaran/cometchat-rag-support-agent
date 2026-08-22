"""
tests/unit/test_policy.py
--------------------------
Unit tests for:
1. Precedence scoring — confirm active/official doc outranks superseded legacy doc.
2. Conflict detection — confirm it fires ONLY on the real active/active conflict
   (docs 11 and 12, dishwasher safety) and NOT on the superseded/active pair
   (docs 01 and 02, returns policy).

All tests run fully offline. No Gemini API calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.models import Chunk
from app.ingestion.parser import parse_directory, parse_file
from app.policy.conflict import ConflictDetector, ConflictGroup, CONFLICT_REGISTRY
from app.policy.scoring import (
    ALPHA,
    BETA,
    BONUS_ACTIVE_OFFICIAL,
    BONUS_CUSTOMER_FACING,
    PENALTY_SUPERSEDED,
    ScoredEvidence,
    is_authoritative,
    score_and_rank,
    score_candidate,
)
from app.retrieval.models import RetrievedCandidate

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_KB_DIR = _REPO_ROOT / "knowledge-base"


# ---------------------------------------------------------------------------
# Helpers to build mock RetrievedCandidates from real chunks
# ---------------------------------------------------------------------------

def _make_candidate(chunk: Chunk, bm25: float = 0.5, dense: float = 0.5) -> RetrievedCandidate:
    return RetrievedCandidate(chunk=chunk, bm25_score=bm25, dense_score=dense)


def _first_chunk(filename: str) -> Chunk:
    """Return the first chunk from a knowledge-base file."""
    return parse_file(_KB_DIR / filename)[0]


# ===========================================================================
# Scoring formula correctness
# ===========================================================================

class TestScoringFormula:
    def test_formula_applied_correctly(self):
        """Verify the formula: final = ALPHA*dense + BETA*bm25 + bonus - penalty."""
        chunk = _first_chunk("01-returns-policy-current.md")
        cand = _make_candidate(chunk, bm25=0.6, dense=0.8)
        ev = score_candidate(cand)
        expected = ALPHA * 0.8 + BETA * 0.6 + ev.metadata_bonus - ev.metadata_penalty
        assert abs(ev.final_score - expected) < 1e-6

    def test_active_official_gets_bonus(self):
        """Active+official doc must have non-zero bonus."""
        chunk = _first_chunk("01-returns-policy-current.md")
        ev = score_candidate(_make_candidate(chunk))
        assert ev.metadata_bonus >= BONUS_ACTIVE_OFFICIAL

    def test_superseded_gets_penalty(self):
        """Superseded doc must have a large penalty applied."""
        chunk = _first_chunk("02-returns-policy-legacy.md")
        ev = score_candidate(_make_candidate(chunk))
        assert ev.metadata_penalty >= PENALTY_SUPERSEDED

    def test_draft_internal_gets_penalty(self):
        """Draft+internal doc must be penalised."""
        chunk = _first_chunk("14-internal-content-migration-notes.md")
        ev = score_candidate(_make_candidate(chunk))
        assert ev.metadata_penalty > 0

    def test_score_components_accessible(self):
        """ScoredEvidence exposes individual components."""
        chunk = _first_chunk("07-warranty.md")
        ev = score_candidate(_make_candidate(chunk, bm25=0.4, dense=0.7))
        assert hasattr(ev, "bm25_score")
        assert hasattr(ev, "dense_score")
        assert hasattr(ev, "metadata_bonus")
        assert hasattr(ev, "metadata_penalty")
        assert hasattr(ev, "final_score")


# ===========================================================================
# Precedence scoring — current vs legacy returns policy
# ===========================================================================

@pytest.fixture(scope="module")
def precedence_ranked():
    """Give both docs equal raw retrieval scores so rank depends purely on metadata."""
    chunk_01 = _first_chunk("01-returns-policy-current.md")
    chunk_02 = _first_chunk("02-returns-policy-legacy.md")
    candidates = [
        _make_candidate(chunk_01, bm25=0.8, dense=0.8),
        _make_candidate(chunk_02, bm25=0.8, dense=0.8),
    ]
    return score_and_rank(candidates)


class TestPrecedenceScoring:
    """
    CORPUS_FACTS.md confirms:
      Doc 01 (RET-2026-01): active, official — the CURRENT policy (30 days)
      Doc 02 (RET-2024-01): superseded, official — the LEGACY policy (45 days)

    For a returns-policy query, the active doc must outscore the legacy doc
    by a significant margin due to the PENALTY_SUPERSEDED applied to doc 02.
    """

    def test_active_doc_ranks_above_superseded(self, precedence_ranked):
        ranked = precedence_ranked
        """Doc 01 (active) must be ranked first, doc 02 (superseded) must be ranked second."""
        assert ranked[0].chunk.metadata.document_id == "RET-2026-01", (
            f"Expected active doc RET-2026-01 to rank first; "
            f"got {ranked[0].chunk.metadata.document_id}"
        )
        assert ranked[1].chunk.metadata.document_id == "RET-2024-01", (
            f"Expected superseded doc RET-2024-01 to rank second"
        )

    def test_score_gap_is_significant(self, precedence_ranked):
        """The gap must be at least PENALTY_SUPERSEDED wide."""
        ranked = precedence_ranked
        gap = ranked[0].final_score - ranked[1].final_score
        assert gap >= PENALTY_SUPERSEDED, (
            f"Score gap {gap:.4f} is less than PENALTY_SUPERSEDED ({PENALTY_SUPERSEDED})"
        )

    def test_active_doc_is_authoritative(self, precedence_ranked):
        """is_authoritative() must return True for the active doc."""
        assert is_authoritative(precedence_ranked[0])

    def test_superseded_doc_not_authoritative(self, precedence_ranked):
        """is_authoritative() must return False for the superseded doc."""
        assert not is_authoritative(precedence_ranked[1])

    def test_rank_order_preserved_across_varied_retrieval_scores(self):
        """Even if legacy doc has higher raw retrieval scores, active doc should win."""
        chunk_01 = _first_chunk("01-returns-policy-current.md")
        chunk_02 = _first_chunk("02-returns-policy-legacy.md")
        candidates = [
            _make_candidate(chunk_01, bm25=0.5, dense=0.5),   # lower raw scores
            _make_candidate(chunk_02, bm25=0.95, dense=0.95),  # higher raw scores
        ]
        ranked = score_and_rank(candidates)
        assert ranked[0].chunk.metadata.document_id == "RET-2026-01", (
            "Active doc must outrank superseded even when superseded has higher retrieval score"
        )


# ===========================================================================
# is_authoritative
# ===========================================================================

class TestIsAuthoritative:
    def test_active_official_customer_is_authoritative(self):
        chunk = _first_chunk("05-domestic-shipping.md")
        ev = score_candidate(_make_candidate(chunk, dense=0.8, bm25=0.8))
        assert is_authoritative(ev)

    def test_superseded_not_authoritative(self):
        chunk = _first_chunk("02-returns-policy-legacy.md")
        ev = score_candidate(_make_candidate(chunk, dense=0.8, bm25=0.8))
        assert not is_authoritative(ev)

    def test_internal_not_authoritative(self):
        chunk = _first_chunk("13-support-escalation.md")
        ev = score_candidate(_make_candidate(chunk, dense=0.8, bm25=0.8))
        assert not is_authoritative(ev)

    def test_draft_not_authoritative(self):
        chunk = _first_chunk("14-internal-content-migration-notes.md")
        ev = score_candidate(_make_candidate(chunk, dense=0.8, bm25=0.8))
        assert not is_authoritative(ev)


# ===========================================================================
# ConflictDetector
# ===========================================================================

class TestConflictDetector:
    """Tests for ConflictDetector.detect()."""

    # --- Helpers ---

    def _evidence_from_file(self, filename: str, bm25: float = 0.8, dense: float = 0.8):
        """Build a list of ScoredEvidence from all chunks in a file."""
        chunks = parse_file(_KB_DIR / filename)
        candidates = [_make_candidate(c, bm25=bm25, dense=dense) for c in chunks]
        return score_and_rank(candidates)

    # --- CONFLICT_REGISTRY sanity ---

    def test_registry_has_exactly_one_confirmed_entry(self):
        """Only one entry must be in the registry (doc 11 vs doc 12)."""
        assert len(CONFLICT_REGISTRY) == 1, (
            f"Expected 1 registry entry; found {len(CONFLICT_REGISTRY)}"
        )

    def test_registry_entry_correct_filenames(self):
        entry = CONFLICT_REGISTRY[0]
        pair = frozenset({entry["doc_a"], entry["doc_b"]})
        assert pair == frozenset({
            "11-product-care.md",
            "12-breeze-tumbler-product-card.md",
        })

    def test_registry_entry_correct_topic(self):
        assert CONFLICT_REGISTRY[0]["topic"] == "breeze_tumbler_dishwasher_safety"

    # --- Detection fires on the real conflict (docs 11 and 12) ---

    def test_conflict_fires_on_dishwasher_pair(self):
        """
        CORPUS_FACTS.md confirmed conflict:
          doc 11 (CARE-2026-01) says body must be hand-washed.
          doc 12 (PROD-BREEZE-20) says all components are dishwasher safe.
          Both are active+official.
        The detector MUST fire when both docs appear in the evidence list.
        """
        ev_11 = self._evidence_from_file("11-product-care.md")
        ev_12 = self._evidence_from_file("12-breeze-tumbler-product-card.md")
        combined = ev_11 + ev_12

        detector = ConflictDetector()
        groups = detector.detect(combined)

        registry_groups = [g for g in groups if g.source == "registry"]
        assert registry_groups, (
            "Expected ConflictDetector to fire on docs 11+12, but no registry conflict found"
        )
        group = registry_groups[0]
        assert group.topic == "breeze_tumbler_dishwasher_safety"
        assert group.confidence == "confirmed"
        filenames = {group.doc_a_filename, group.doc_b_filename}
        assert filenames == {
            "11-product-care.md",
            "12-breeze-tumbler-product-card.md",
        }

    def test_conflict_fires_regardless_of_query_phrasing(self):
        """
        The registry is keyed on filenames, not query text, so it must fire
        even when the evidence was retrieved via a very differently-worded query.
        We simulate this by directly providing evidence from both files.
        """
        # Evidence retrieved via a hypothetical "cleaning care bottle" query
        ev_11 = self._evidence_from_file("11-product-care.md")
        ev_12 = self._evidence_from_file("12-breeze-tumbler-product-card.md")
        combined = ev_11 + ev_12

        detector = ConflictDetector()
        groups = detector.detect(combined)
        assert any(
            g.source == "registry" and "breeze_tumbler" in g.topic
            for g in groups
        ), "Conflict should fire regardless of how the query was phrased"

    # --- Detection does NOT fire on the superseded/active pair (docs 01 and 02) ---

    def test_no_conflict_on_returns_policy_pair(self):
        """
        Docs 01 (active) and 02 (superseded) must NOT trigger a registry conflict.
        Doc 02 is superseded, so ConflictDetector only considers active+official chunks.
        The detector must return zero registry-sourced conflicts for this pair.
        """
        ev_01 = self._evidence_from_file("01-returns-policy-current.md")
        ev_02 = self._evidence_from_file("02-returns-policy-legacy.md")
        combined = ev_01 + ev_02

        detector = ConflictDetector()
        groups = detector.detect(combined)

        registry_groups = [g for g in groups if g.source == "registry"]
        assert not registry_groups, (
            f"Conflict detector must NOT fire on docs 01+02 (superseded pair); "
            f"got: {[g.topic for g in registry_groups]}"
        )

    def test_no_conflict_on_single_document(self):
        """A single document in evidence cannot conflict with itself."""
        ev = self._evidence_from_file("01-returns-policy-current.md")
        detector = ConflictDetector()
        groups = detector.detect(ev)
        assert not groups, f"No conflict expected for single document; got: {groups}"

    def test_no_conflict_on_empty_evidence(self):
        """Empty evidence list should produce no conflicts."""
        detector = ConflictDetector()
        groups = detector.detect([])
        assert groups == []

    def test_conflict_group_has_required_fields(self):
        """ConflictGroup must expose all required fields."""
        ev_11 = self._evidence_from_file("11-product-care.md")
        ev_12 = self._evidence_from_file("12-breeze-tumbler-product-card.md")
        detector = ConflictDetector()
        groups = detector.detect(ev_11 + ev_12)
        assert groups
        g = groups[0]
        assert g.topic
        assert g.doc_a_filename
        assert g.doc_b_filename
        assert isinstance(g.doc_a_chunk, Chunk)
        assert isinstance(g.doc_b_chunk, Chunk)
        assert g.note
        assert g.source in ("registry", "numeric_fallback")
        assert g.confidence in ("confirmed", "tentative")

    # --- score_and_rank with full corpus ---

    def test_full_corpus_scoring_has_no_internal_doc_at_top(self):
        """
        When all chunks are scored at equal retrieval scores, internal/draft
        docs must not rank at the top.
        """
        all_chunks = parse_directory(_KB_DIR)
        candidates = [_make_candidate(c, bm25=0.5, dense=0.5) for c in all_chunks]
        ranked = score_and_rank(candidates)

        # The top-ranked chunk must be from an active+official+customer doc
        top = ranked[0]
        assert top.chunk.metadata.status == "active", (
            f"Top chunk is not active: {top.chunk.metadata.filename}"
        )
        assert top.chunk.metadata.policy_authority == "official", (
            f"Top chunk is not official: {top.chunk.metadata.filename}"
        )
        assert top.chunk.metadata.audience == "customer", (
            f"Top chunk is not customer-facing: {top.chunk.metadata.filename}"
        )


# ===========================================================================
# End-to-end: retrieval + scoring + conflict (no network — uses pre-built chunks)
# ===========================================================================

class TestEndToEndPolicyWithRealChunks:
    """
    Simulate what the agent does on the dishwasher query end-to-end,
    bypassing the embedding model to stay fully offline.
    """

    def test_dishwasher_query_active_doc_ranked_above_any_superseded(self):
        """
        Combine chunks from all relevant documents with realistic scores
        and confirm the active doc ranks above any superseded doc.
        """
        chunk_01 = _first_chunk("01-returns-policy-current.md")
        chunk_02 = _first_chunk("02-returns-policy-legacy.md")
        chunk_11 = _first_chunk("11-product-care.md")
        chunk_12 = _first_chunk("12-breeze-tumbler-product-card.md")

        candidates = [
            _make_candidate(chunk_11, bm25=0.9, dense=0.9),
            _make_candidate(chunk_12, bm25=0.85, dense=0.88),
            _make_candidate(chunk_01, bm25=0.3, dense=0.2),
            _make_candidate(chunk_02, bm25=0.3, dense=0.2),
        ]
        ranked = score_and_rank(candidates)

        # Doc 02 (superseded) must rank below all active docs
        doc02_rank = next(
            i for i, e in enumerate(ranked)
            if e.chunk.metadata.document_id == "RET-2024-01"
        )
        active_ranks = [
            i for i, e in enumerate(ranked)
            if e.chunk.metadata.status == "active"
        ]
        assert all(doc02_rank > r for r in active_ranks), (
            "Superseded doc 02 must rank below all active docs"
        )

    def test_conflict_detection_on_dishwasher_evidence(self):
        """
        With dishwasher-related evidence, ConflictDetector must flag docs 11+12.
        """
        chunk_11 = _first_chunk("11-product-care.md")
        chunk_12 = _first_chunk("12-breeze-tumbler-product-card.md")
        candidates = [
            _make_candidate(chunk_11, bm25=0.9, dense=0.9),
            _make_candidate(chunk_12, bm25=0.85, dense=0.88),
        ]
        evidence = score_and_rank(candidates)

        detector = ConflictDetector()
        groups = detector.detect(evidence)

        assert groups, "No conflict detected for docs 11+12"
        assert groups[0].source == "registry"
        assert groups[0].confidence == "confirmed"
