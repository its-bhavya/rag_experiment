"""Hybrid retrieval fusing sparse and dense rankings via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from src.retrieval.base import BaseRetriever, ScoredDoc

logger = logging.getLogger(__name__)

DEFAULT_RRF_C = 60


def reciprocal_rank_fusion(
    system_rankings: Sequence[Sequence[ScoredDoc]],
    c: int = DEFAULT_RRF_C,
    k: int = 10,
) -> List[ScoredDoc]:
    r"""Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Definition:
        For a set of retrieval systems S and a document d:
            \text{RRF}(d) = \sum_{s \in S} \frac{1}{c + \text{rank}_s(d)}
        where:
            - c is a smoothing constant (standard default c=60).
            - \text{rank}_s(d) is the 1-indexed rank of d in system s.

    Missing Document Handling:
        If a document d is retrieved by system A but is absent from system B's candidate list,
        system B contributes 0.0 to d's score. Documents retrieved by multiple systems accumulate
        fused scores from all systems where they appear.

    Deduplication:
        Document IDs are aggregated into a unique score map, guaranteeing no duplicate document IDs.

    Args:
        system_rankings: Sequence of ranked candidate lists (each a Sequence of ScoredDoc).
        c: Smoothing parameter (default 60).
        k: Maximum number of fused documents to return.

    Returns:
        List of ScoredDoc sorted by descending RRF score, with 1-indexed ranks.
    """
    if c <= 0:
        raise ValueError(f"c must be a positive integer, got c={c}")
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got k={k}")

    rrf_scores: Dict[str, float] = {}

    for ranking in system_rankings:
        for scored_doc in ranking:
            doc_id = scored_doc.doc_id
            rank = scored_doc.rank
            if rank < 1:
                raise ValueError(f"Document rank must be 1-indexed, got {rank}")

            rrf_score = 1.0 / (c + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

    # Deterministic sorting: sort primarily by RRF score descending,
    # secondarily by doc_id ascending for reproducible tie-breaking
    sorted_items = sorted(rrf_scores.items(), key=lambda item: (-item[1], item[0]))

    fused_ranking: List[ScoredDoc] = []
    for rank, (doc_id, score) in enumerate(sorted_items[:k], start=1):
        fused_ranking.append(
            ScoredDoc(
                doc_id=doc_id,
                score=float(score),
                rank=rank,
            )
        )

    return fused_ranking


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining Sparse (BM25) and Dense (Sentence Transformers) via RRF."""

    def __init__(
        self,
        sparse_retriever: BaseRetriever,
        dense_retriever: BaseRetriever,
        c: int = DEFAULT_RRF_C,
        candidate_pool_size: int = 100,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            sparse_retriever: BaseRetriever instance (e.g. BM25Retriever).
            dense_retriever: BaseRetriever instance (e.g. DenseRetriever).
            c: RRF smoothing constant (default 60).
            candidate_pool_size: Number of candidates to retrieve from each system before fusion.
        """
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.c = c
        self.candidate_pool_size = candidate_pool_size

    def retrieve(self, query: str, k: int = 10) -> List[ScoredDoc]:
        """Retrieve top-K fused documents for query using RRF over sparse and dense candidates.

        Args:
            query: Query string.
            k: Top-K documents to return.

        Returns:
            List of ScoredDoc ordered by descending RRF score.
        """
        # Retrieve candidate lists from both systems
        fetch_k = max(k, self.candidate_pool_size)
        sparse_ranking = self.sparse_retriever.retrieve(query, k=fetch_k)
        dense_ranking = self.dense_retriever.retrieve(query, k=fetch_k)

        # Fuse rankings using Reciprocal Rank Fusion
        return reciprocal_rank_fusion(
            system_rankings=[sparse_ranking, dense_ranking],
            c=self.c,
            k=k,
        )
