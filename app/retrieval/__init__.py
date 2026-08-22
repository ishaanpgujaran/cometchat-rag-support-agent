"""app/retrieval — hybrid BM25 + dense retrieval."""

from app.retrieval.models import RetrievedCandidate
from app.retrieval.index import HybridIndex, hybrid_search, get_default_index

__all__ = [
    "RetrievedCandidate",
    "HybridIndex",
    "hybrid_search",
    "get_default_index",
]
