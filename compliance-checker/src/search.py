"""
search.py
---------
Builds and queries a FAISS vector index over design document paragraphs.
For each standard clause, retrieves the top-K most semantically similar
paragraphs from the design document.

Algorithm
---------
1. The design document is split into paragraphs.
2. Each paragraph is encoded into a 384-dim L2-normalised vector (MiniLM-L6).
3. A FAISS IndexFlatIP (exact inner-product) index is built over all vectors.
4. For each compliance clause:
   a. Encode the clause text → query vector.
   b. FAISS.search(query, k=3) returns top-3 most similar paragraphs.
   c. Return paragraph text + cosine similarity score.

Why FAISS?
----------
For 2,000 paragraphs and 50 clauses, brute-force cosine similarity requires
100,000 × 384-dim dot products. FAISS IndexFlatIP does this in <50ms on CPU
using BLAS-accelerated matrix multiplication.

Usage:
    from src.search import DocumentIndex
    from src.embedder import Embedder

    emb   = Embedder()
    index = DocumentIndex(emb)
    index.build(["paragraph one...", "paragraph two...", ...])

    matches = index.query("traces carrying >1A shall be 0.3mm minimum", top_k=3)
    # → [{"text": "gate driver conductors are 1.5mm, rated 15A", "score": 0.91, "idx": 42}, ...]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import faiss
import numpy as np

from src.embedder import Embedder, get_embedder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single matching paragraph returned by a vector search query."""
    text      : str
    score     : float          # Cosine similarity in [0, 1]
    paragraph_idx: int         # Index in the original paragraph list
    is_match  : bool = field(init=False)

    def __post_init__(self):
        # Semantic match threshold: cosine similarity ≥ 0.65
        self.is_match = self.score >= 0.65

    def confidence_pct(self) -> int:
        """Map cosine similarity → confidence percentage for the UI."""
        return min(100, int(self.score * 110))


class DocumentIndex:
    """
    Manages the FAISS index for one design document.

    Lifecycle
    ---------
    build()  → encode paragraphs, build FAISS index
    query()  → find top-K matching paragraphs for a clause
    reset()  → clear the index (e.g. when a new document is uploaded)
    """

    # Cosine similarity thresholds (for L2-normalised vectors, dot product = cosine)
    STRONG_MATCH  = 0.80   # → PASS (confident match)
    PARTIAL_MATCH = 0.65   # → WARN (some evidence, not conclusive)
    NO_MATCH      = 0.50   # → FAIL (insufficient evidence)

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        embedding_dim: int = 384,
    ) -> None:
        self.embedder      = embedder or get_embedder()
        self.embedding_dim = embedding_dim
        self._index: Optional[faiss.IndexFlatIP] = None
        self._paragraphs: List[str] = []
        self._embeddings: Optional[np.ndarray] = None

    # ── Build ──────────────────────────────────────────────────────────────
    def build(self, paragraphs: List[str], batch_size: int = 64) -> None:
        """
        Encode a list of document paragraphs and build the FAISS index.

        Parameters
        ----------
        paragraphs : list[str]
            All paragraphs extracted from the design document.
        batch_size : int
            Batch size for the embedding model.
        """
        if not paragraphs:
            raise ValueError("paragraphs list is empty — nothing to index.")

        logger.info(f"Building FAISS index over {len(paragraphs)} paragraphs ...")

        self._paragraphs = paragraphs
        self._embeddings = self.embedder.encode(
            paragraphs,
            batch_size=batch_size,
            normalise=True,
        )                                           # shape: (n, 384)

        # IndexFlatIP: exact inner product search on L2-normalised vectors
        # = exact cosine similarity, no approximation.
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._index.add(self._embeddings)           # add all paragraph vectors

        logger.info(
            f"FAISS index built — {self._index.ntotal} vectors, "
            f"dim={self.embedding_dim}"
        )

    # ── Query ──────────────────────────────────────────────────────────────
    def query(self, clause_text: str, top_k: int = 3) -> List[SearchResult]:
        """
        Find the top-K document paragraphs most semantically similar to a
        compliance clause.

        Parameters
        ----------
        clause_text : str
            The requirement clause text (e.g. "traces carrying >1A shall be 0.3mm").
        top_k : int
            Number of results to return (default 3).

        Returns
        -------
        list[SearchResult]
            Sorted by similarity score descending.
        """
        if self._index is None:
            raise RuntimeError(
                "Index not built yet. Call build(paragraphs) first."
            )

        k = min(top_k, len(self._paragraphs))

        # Encode and normalise the query clause
        q_vec = self.embedder.encode(clause_text, normalise=True)   # (384,)
        q_vec = q_vec.reshape(1, -1).astype(np.float32)             # (1, 384)

        # FAISS search — returns distances (inner products) and indices
        distances, indices = self._index.search(q_vec, k)           # (1,k), (1,k)

        results: List[SearchResult] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue                                             # FAISS padding
            results.append(SearchResult(
                text=self._paragraphs[idx],
                score=float(score),
                paragraph_idx=int(idx),
            ))

        return sorted(results, key=lambda r: r.score, reverse=True)

    # ── Helpers ────────────────────────────────────────────────────────────
    def best_match(self, clause_text: str) -> Optional[SearchResult]:
        """Return the single best matching paragraph, or None if no match."""
        results = self.query(clause_text, top_k=1)
        return results[0] if results else None

    def threshold_label(self, score: float) -> str:
        """Map a cosine similarity score to a human-readable label."""
        if score >= self.STRONG_MATCH:
            return "strong_match"
        elif score >= self.PARTIAL_MATCH:
            return "partial_match"
        else:
            return "no_match"

    def reset(self) -> None:
        """Clear the index. Call before loading a new document."""
        self._index      = None
        self._paragraphs = []
        self._embeddings = None
        logger.info("FAISS index cleared.")

    @property
    def is_built(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def paragraph_count(self) -> int:
        return len(self._paragraphs)
