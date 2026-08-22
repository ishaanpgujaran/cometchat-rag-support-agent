"""
app/retrieval/models.py
-----------------------
Data models shared across retrieval and policy layers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.ingestion.models import Chunk


class RetrievedCandidate(BaseModel):
    """A chunk returned by the hybrid retrieval system with both component scores."""

    chunk: Chunk
    bm25_score: float = Field(description="Normalised BM25 score in [0, 1]")
    dense_score: float = Field(description="Cosine similarity score in [0, 1] after clipping to [0, 1]")
