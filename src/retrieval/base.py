"""Base interface and data structures for retrieval systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class ScoredDoc:
    """Represents a single retrieved document candidate.

    Attributes:
        doc_id: The unique identifier of the retrieved document.
        score: The retrieval score (BM25 score, cosine similarity, or RRF score).
        rank: 1-indexed position in the ranked candidate list (1 = top rank).
    """

    doc_id: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"Rank must be 1-indexed (>= 1), got rank={self.rank}")


class BaseRetriever(ABC):
    """Abstract base class for all retrieval methods (BM25, Dense, Hybrid)."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[ScoredDoc]:
        """Retrieve top-K ranked documents for a given query string.

        Args:
            query: Query text string.
            k: Number of documents to retrieve (must be >= 1).

        Returns:
            List of ScoredDoc ordered by descending score with 1-indexed ranks.
        """
        pass

    def retrieve_batch(
        self, queries: Sequence[str], k: int = 10
    ) -> List[List[ScoredDoc]]:
        """Retrieve top-K documents for a batch of queries.

        Args:
            queries: Sequence of query text strings.
            k: Number of documents to retrieve per query.

        Returns:
            List of ranked ScoredDoc lists, corresponding to each input query.
        """
        return [self.retrieve(q, k=k) for q in queries]
