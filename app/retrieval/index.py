"""
app/retrieval/index.py
----------------------
BM25 and dense (sentence-transformers) retrieval indices with hybrid search.

Design decisions:
- BM25: uses rank_bm25.BM25Okapi, tokenised by simple whitespace+lowercasing.
  Scores are normalised to [0, 1] by dividing by the max score in the result set
  (or 1.0 if the max is 0 to avoid division by zero).
- Dense: uses sentence-transformers with the model name from app.config.EMBEDDING_MODEL
  (default: "BAAI/bge-small-en-v1.5"). Similarity is cosine, clipped to [0, 1].
- Embedding cache: embeddings are persisted to `embeddings_cache/embeddings.pkl`
  under the project root so they are not recomputed on every run.
  The cache is keyed on (model_name, list_of_chunk_ids) so it is invalidated
  automatically if the corpus or model changes.
  The cache file is gitignored (see .gitignore: embeddings_cache/).
- Hybrid search: combines both scores as
      hybrid = ALPHA * dense_score + (1 - ALPHA) * bm25_score
  using app.policy.scoring constants. Because policy scoring is downstream,
  retrieval just returns RetrievedCandidate with both raw scores.
  The merger uses the union of BM25 and dense result sets (top-k*2 from each
  before merging) so every re-ranked candidate has both scores available.
- No Gemini API calls anywhere in this module.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, KNOWLEDGE_BASE_DIR
from app.ingestion.models import Chunk
from app.ingestion.parser import parse_directory
from app.retrieval.models import RetrievedCandidate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _PROJECT_ROOT / "embeddings_cache"
_CACHE_FILE = _CACHE_DIR / "embeddings.pkl"


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9']+", text.lower())


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _corpus_fingerprint(chunks: list[Chunk], model_name: str) -> str:
    """A short hash that changes when the corpus or model changes."""
    ids = "||".join(c.chunk_id for c in chunks)
    raw = f"{model_name}::{ids}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_cache(
    chunks: list[Chunk], model_name: str
) -> Optional[np.ndarray]:
    """Return cached embeddings if the cache is valid, else None."""
    if not _CACHE_FILE.exists():
        return None
    try:
        with _CACHE_FILE.open("rb") as f:
            cached = pickle.load(f)
        if cached.get("fingerprint") != _corpus_fingerprint(chunks, model_name):
            logger.info("Embedding cache fingerprint mismatch — will recompute.")
            return None
        return cached["embeddings"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load embedding cache (%s) — will recompute.", exc)
        return None


def _save_cache(chunks: list[Chunk], model_name: str, embeddings: np.ndarray) -> None:
    """Persist embeddings to disk."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": _corpus_fingerprint(chunks, model_name),
        "embeddings": embeddings,
    }
    with _CACHE_FILE.open("wb") as f:
        pickle.dump(payload, f)
    logger.info("Embedding cache saved to %s", _CACHE_FILE)


# ---------------------------------------------------------------------------
# Main index class
# ---------------------------------------------------------------------------

class HybridIndex:
    """Holds both a BM25 index and a dense embedding index over a corpus.

    Parameters
    ----------
    chunks:
        The full list of Chunk objects to index.
    model_name:
        Sentence-transformers model identifier. Defaults to ``EMBEDDING_MODEL``
        from app.config.
    cache_path:
        Override the default cache file location (useful in tests).
    """

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = EMBEDDING_MODEL,
        cache_path: Optional[Path] = None,
    ) -> None:
        self._chunks = chunks
        self._model_name = model_name
        self._cache_path = cache_path or _CACHE_FILE

        # BM25
        tokenised = [_tokenise(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenised)

        # Dense
        self._st_model = SentenceTransformer(model_name)
        self._corpus_embeddings = self._build_or_load_embeddings(chunks, model_name)

    # ------------------------------------------------------------------
    # Embedding management
    # ------------------------------------------------------------------

    def _build_or_load_embeddings(
        self, chunks: list[Chunk], model_name: str
    ) -> np.ndarray:
        cache_file = self._cache_path
        cache_dir = cache_file.parent

        # Try loading from cache first
        if cache_file.exists():
            try:
                with cache_file.open("rb") as f:
                    cached = pickle.load(f)
                fp = _corpus_fingerprint(chunks, model_name)
                if cached.get("fingerprint") == fp:
                    logger.info("Loaded %d embeddings from cache.", len(chunks))
                    return cached["embeddings"]
                logger.info("Cache fingerprint mismatch — recomputing.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cache load failed (%s) — recomputing.", exc)

        logger.info("Computing embeddings for %d chunks with model '%s'.", len(chunks), model_name)
        texts = [c.text for c in chunks]
        embeddings = self._st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)

        # Persist
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": _corpus_fingerprint(chunks, model_name),
            "embeddings": embeddings,
        }
        with cache_file.open("wb") as f:
            pickle.dump(payload, f)
        logger.info("Embeddings cached to %s", cache_file)

        return embeddings

    # ------------------------------------------------------------------
    # BM25 retrieval
    # ------------------------------------------------------------------

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return list of (chunk_index, normalised_score) sorted descending."""
        q_tokens = _tokenise(query)
        raw_scores = self._bm25.get_scores(q_tokens)
        max_score = float(raw_scores.max()) if raw_scores.max() > 0 else 1.0
        normed = raw_scores / max_score

        top_indices = np.argsort(normed)[::-1][:k]
        return [(int(i), float(normed[i])) for i in top_indices if normed[i] > 0]

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return list of (chunk_index, cosine_similarity) sorted descending.

        Cosine similarity is clipped to [0, 1] (negative values indicate
        anti-correlated content and are treated as 0).
        """
        q_emb = self._st_model.encode([query], convert_to_numpy=True).astype(np.float32)
        # Normalise corpus and query for cosine similarity
        corpus_norm = self._corpus_embeddings / (
            np.linalg.norm(self._corpus_embeddings, axis=1, keepdims=True) + 1e-10
        )
        q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
        scores = (corpus_norm @ q_norm.T).flatten()
        scores = np.clip(scores, 0.0, 1.0)

        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    # ------------------------------------------------------------------
    # Hybrid search (public API)
    # ------------------------------------------------------------------

    def hybrid_search(self, query: str, k: int) -> list[RetrievedCandidate]:
        """Return up to *k* RetrievedCandidate objects, merged from BM25+dense.

        Strategy:
        - Fetch top 2k results from each index (so the union is large enough).
        - Merge into a dict keyed on chunk index, assigning score 0 for any
          system that did not return a given chunk.
        - Sort by (bm25_score + dense_score) descending, take top k.
          Final policy scoring (ALPHA/BETA weighted) happens in app/policy/scoring.
        """
        pool = k * 2
        bm25_results = dict(self.bm25_search(query, pool))
        dense_results = dict(self.dense_search(query, pool))

        all_indices = set(bm25_results) | set(dense_results)
        candidates: list[RetrievedCandidate] = []
        for idx in all_indices:
            b_score = bm25_results.get(idx, 0.0)
            d_score = dense_results.get(idx, 0.0)
            candidates.append(
                RetrievedCandidate(
                    chunk=self._chunks[idx],
                    bm25_score=b_score,
                    dense_score=d_score,
                )
            )

        # Pre-sort by naive sum before handing off to policy scorer
        candidates.sort(key=lambda c: c.bm25_score + c.dense_score, reverse=True)
        return candidates[:k]


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------

_default_index: Optional[HybridIndex] = None


def get_default_index(knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR) -> HybridIndex:
    """Return (and lazily build) the default global HybridIndex.

    The index is built once per process from all .md files in
    ``knowledge_base_dir``.  Subsequent calls return the same object.
    """
    global _default_index
    if _default_index is None:
        chunks = parse_directory(knowledge_base_dir)
        _default_index = HybridIndex(chunks)
    return _default_index


def hybrid_search(
    query: str,
    k: int,
    knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR,
) -> list[RetrievedCandidate]:
    """Module-level convenience function using the default index."""
    return get_default_index(knowledge_base_dir).hybrid_search(query, k)
