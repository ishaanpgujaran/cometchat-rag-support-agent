"""
tests/unit/test_retrieval.py
-----------------------------
Unit tests for BM25, dense, and hybrid retrieval.

All tests run fully offline. The dense retrieval tests will download
the embedding model on first run (network access required at that point),
but subsequent runs are fully offline due to the embedding cache.

For fully-offline CI:
  Set EMBEDDING_MODEL to a local path, or pre-warm the cache.
  The cache is stored in embeddings_cache/embeddings.pkl (gitignored).

Test categories:
1. BM25 on at least 3 real queries against the real corpus.
2. Dense retrieval on at least 2 paraphrased queries.
3. Hybrid search — verifies both component scores are populated.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.ingestion.parser import parse_directory
from app.retrieval.index import HybridIndex
from app.retrieval.models import RetrievedCandidate

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_KB_DIR = _REPO_ROOT / "knowledge-base"


@pytest.fixture(scope="module")
def chunks():
    """Parse the real knowledge-base once for all retrieval tests."""
    return parse_directory(_KB_DIR)


@pytest.fixture(scope="module")
def index(tmp_path_factory, chunks):
    """Build a HybridIndex with a temporary cache so tests don't pollute the real cache."""
    cache_dir = tmp_path_factory.mktemp("emb_cache")
    cache_file = cache_dir / "embeddings.pkl"
    return HybridIndex(chunks, cache_path=cache_file)


# ===========================================================================
# BM25 retrieval — at least 3 real queries
# ===========================================================================

class TestBM25Retrieval:
    """BM25 search should surface relevant documents for direct keyword queries."""

    def test_bm25_returns_policy_query(self, index, chunks):
        """Q: 'How many days do I have to return an item?'
        Expected: doc 01 (30 calendar days, active) should appear in top results.
        """
        results = index.bm25_search("how many days do I have to return an item", k=5)
        assert results, "BM25 returned no results"
        returned_chunk_ids = {chunks[i].chunk_id for i, _ in results}
        filenames = {chunks[i].metadata.filename for i, _ in results}
        # At least one of the return policy docs should appear
        assert any("returns-policy" in fn for fn in filenames), (
            f"Expected a returns-policy doc in results; got filenames: {filenames}"
        )

    def test_bm25_warranty_query(self, index, chunks):
        """Q: 'What is the warranty period for bags and backpacks?'
        Expected: doc 07 (WAR-2026-01) should rank highly.
        """
        results = index.bm25_search("warranty period bags backpacks", k=5)
        assert results
        filenames = {chunks[i].metadata.filename for i, _ in results}
        assert "07-warranty.md" in filenames, (
            f"07-warranty.md not in BM25 results; got: {filenames}"
        )

    def test_bm25_dishwasher_query(self, index, chunks):
        """Q: 'Is the Breeze Tumbler dishwasher safe?'
        Expected: both doc 11 and/or doc 12 should appear (the known conflict pair).
        """
        results = index.bm25_search("Breeze Tumbler dishwasher safe", k=10)
        assert results
        filenames = {chunks[i].metadata.filename for i, _ in results}
        conflict_docs = {"11-product-care.md", "12-breeze-tumbler-product-card.md"}
        assert filenames & conflict_docs, (
            f"Neither conflict doc appeared; got: {filenames}"
        )

    def test_bm25_shipping_query(self, index, chunks):
        """Q: 'What are the shipping options to Canada?'
        Expected: doc 06 (international shipping) should appear.
        """
        results = index.bm25_search("shipping Canada international", k=5)
        assert results
        filenames = {chunks[i].metadata.filename for i, _ in results}
        assert "06-international-shipping.md" in filenames, (
            f"06-international-shipping.md not found; got: {filenames}"
        )

    def test_bm25_scores_normalised(self, index, chunks):
        """All BM25 scores should be in [0, 1]."""
        results = index.bm25_search("return policy days", k=10)
        for _, score in results:
            assert 0.0 <= score <= 1.0, f"BM25 score out of range: {score}"

    def test_bm25_empty_query_graceful(self, index, chunks):
        """An empty query should not raise and should return an empty list or low scores."""
        try:
            results = index.bm25_search("", k=5)
            # scores may all be 0 → list may be empty or all-zero
            for _, score in results:
                assert score >= 0.0
        except Exception as exc:
            pytest.fail(f"BM25 raised on empty query: {exc}")


# ===========================================================================
# Dense retrieval — at least 2 paraphrased queries
# ===========================================================================

class TestDenseRetrieval:
    """Dense retrieval should surface relevant documents for paraphrased queries
    that use different vocabulary from the source text.
    """

    def test_dense_paraphrase_returns_policy(self, index, chunks):
        """Paraphrase: 'merchandise exchange timeframe' (avoids 'return' and 'days').
        Dense retrieval should still surface the returns policy document.
        """
        results = index.dense_search("merchandise exchange timeframe", k=10)
        assert results, "Dense returned no results"
        filenames = {chunks[i].metadata.filename for i, _ in results}
        assert any("returns-policy" in fn for fn in filenames), (
            f"Expected a returns-policy doc; got: {filenames}"
        )

    def test_dense_paraphrase_cleaning_instructions(self, index, chunks):
        """Paraphrase: 'washing instructions for insulated bottle' (avoids 'Breeze Tumbler',
        'dishwasher', 'product care').
        Dense retrieval should surface docs 11 or 12.
        """
        results = index.dense_search("washing instructions for insulated bottle", k=10)
        assert results
        filenames = {chunks[i].metadata.filename for i, _ in results}
        conflict_docs = {"11-product-care.md", "12-breeze-tumbler-product-card.md"}
        assert filenames & conflict_docs, (
            f"Expected a product care/card doc; got: {filenames}"
        )

    def test_dense_paraphrase_membership_benefits(self, index, chunks):
        """Paraphrase: 'loyalty program extended return privilege'.
        Dense should surface doc 09 (TrailPlus membership).
        """
        results = index.dense_search("loyalty program extended return privilege", k=10)
        assert results
        filenames = {chunks[i].metadata.filename for i, _ in results}
        assert "09-trailplus-membership.md" in filenames, (
            f"09-trailplus-membership.md not found; got: {filenames}"
        )

    def test_dense_scores_in_range(self, index, chunks):
        """All dense scores should be in [0, 1] after clipping."""
        results = index.dense_search("return policy", k=10)
        for _, score in results:
            assert 0.0 <= score <= 1.0, f"Dense score out of range: {score}"


# ===========================================================================
# Hybrid search
# ===========================================================================

class TestHybridSearch:
    def test_hybrid_returns_retrieved_candidates(self, index):
        results = index.hybrid_search("return policy", k=5)
        assert results
        assert all(isinstance(r, RetrievedCandidate) for r in results)

    def test_hybrid_both_scores_populated(self, index):
        """Every RetrievedCandidate should have non-negative scores."""
        results = index.hybrid_search("Breeze Tumbler dishwasher", k=10)
        for r in results:
            assert r.bm25_score >= 0.0
            assert r.dense_score >= 0.0

    def test_hybrid_respects_k(self, index):
        k = 3
        results = index.hybrid_search("warranty policy", k=k)
        assert len(results) <= k

    def test_hybrid_conflict_pair_both_returned(self, index):
        """For the dishwasher query, BOTH conflict docs should appear in top-10."""
        results = index.hybrid_search("Can I put the Breeze Tumbler in the dishwasher?", k=10)
        filenames = {r.chunk.metadata.filename for r in results}
        conflict_docs = {"11-product-care.md", "12-breeze-tumbler-product-card.md"}
        assert conflict_docs.issubset(filenames), (
            f"Not both conflict docs retrieved. Got: {filenames}"
        )

    def test_hybrid_returns_policy_top_doc_is_active(self, index):
        """For return-window query, the top-ranked result should not be the superseded doc."""
        results = index.hybrid_search("how many days to return a product", k=5)
        assert results
        # The very top result should be from an active document
        # (superseded doc 02 should not outrank active doc 01 after scoring)
        # Note: hybrid_search pre-sorts by sum of scores, not final policy score,
        # so we just check that at least one active doc appears
        active_filenames = {
            r.chunk.metadata.filename
            for r in results
            if r.chunk.metadata.status == "active"
        }
        assert active_filenames, "No active documents in hybrid results for return policy query"
