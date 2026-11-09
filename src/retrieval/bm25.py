"""BM25 sparse retrieval implementation using rank_bm25."""

from __future__ import annotations

import logging
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import numpy as np
from rank_bm25 import BM25Okapi

from src.datasets.scifact import SciFactDocument
from src.retrieval.base import BaseRetriever, ScoredDoc

logger = logging.getLogger(__name__)


def tokenize_bm25(text: str) -> List[str]:
    """Tokenize and preprocess text for BM25.

    Normalizes unicode, lowercases, and extracts alphanumeric/word tokens.
    Handles empty text, punctuation, and scientific terms gracefully.
    """
    if not text:
        return []
    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)
    # Alphanumeric token extraction
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
    # Strip leading/trailing hyphens from tokens
    clean = [t.strip("-") for t in tokens if t.strip("-")]
    return clean


class BM25Retriever(BaseRetriever):
    """BM25 Okapi retriever over corpus documents."""

    def __init__(
        self,
        doc_ids: Sequence[str],
        doc_texts: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize BM25 retriever.

        Args:
            doc_ids: List of unique document identifiers.
            doc_texts: Corresponding document texts (title + text).
            k1: BM25 k1 parameter (term frequency saturation). Default 1.5.
            b: BM25 b parameter (document length normalization). Default 0.75.
        """
        if len(doc_ids) != len(doc_texts):
            raise ValueError(
                f"Mismatch: len(doc_ids)={len(doc_ids)} vs len(doc_texts)={len(doc_texts)}"
            )

        self.doc_ids = list(doc_ids)
        self.k1 = k1
        self.b = b

        logger.info(f"Tokenizing {len(doc_texts)} documents for BM25...")
        self.tokenized_corpus = [tokenize_bm25(t) for t in doc_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        logger.info("BM25 index built successfully.")

    @classmethod
    def from_corpus(
        cls,
        corpus: Mapping[str, SciFactDocument | Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Retriever:
        """Factory method to construct retriever from a SciFact corpus dictionary."""
        doc_ids: List[str] = []
        doc_texts: List[str] = []

        for doc_id, doc in corpus.items():
            doc_ids.append(str(doc_id))
            if isinstance(doc, SciFactDocument):
                full_text = doc.full_text
            elif isinstance(doc, dict):
                title = doc.get("title", "")
                text = doc.get("text", "")
                full_text = f"{title}\n{text}".strip() if title else text.strip()
            else:
                full_text = str(doc)
            doc_texts.append(full_text)

        return cls(doc_ids=doc_ids, doc_texts=doc_texts, k1=k1, b=b)

    def retrieve(self, query: str, k: int = 10) -> List[ScoredDoc]:
        """Retrieve top-K ranked documents for query using BM25.

        Args:
            query: Query text string.
            k: Number of documents to retrieve.

        Returns:
            List of ScoredDoc ordered by descending score with 1-indexed ranks.
        """
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got k={k}")

        q_tokens = tokenize_bm25(query)
        if not q_tokens:
            # Query has no valid alphanumeric tokens; return top k documents with zero score
            logger.warning(f"Query has no valid tokens: '{query}'")
            return [
                ScoredDoc(doc_id=self.doc_ids[i], score=0.0, rank=i + 1)
                for i in range(min(k, len(self.doc_ids)))
            ]

        scores = self.bm25.get_scores(q_tokens)

        # Deterministic sorting: sort by score DESC, then doc_id ASC for exact tie-breaking.
        # Use stable Python sort keyed on (-score, doc_id) instead of np.argsort which
        # has non-deterministic tie-breaking behavior.
        num_candidates = min(k, len(self.doc_ids))
        indexed_scores = sorted(
            range(len(self.doc_ids)),
            key=lambda i: (-scores[i], self.doc_ids[i]),
        )
        top_indices = indexed_scores[:num_candidates]

        ranked_docs: List[ScoredDoc] = []
        for rank, idx in enumerate(top_indices, start=1):
            ranked_docs.append(
                ScoredDoc(
                    doc_id=self.doc_ids[idx],
                    score=float(scores[idx]),
                    rank=rank,
                )
            )

        return ranked_docs

    def save(self, file_path: Path | str) -> None:
        """Serialize BM25 index and document IDs to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "doc_ids": self.doc_ids,
                    "tokenized_corpus": self.tokenized_corpus,
                    "k1": self.k1,
                    "b": self.b,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info(f"Saved BM25 index to {path}")

    @classmethod
    def load(cls, file_path: Path | str) -> BM25Retriever:
        """Load serialized BM25 index from disk."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        instance.doc_ids = data["doc_ids"]
        instance.tokenized_corpus = data["tokenized_corpus"]
        instance.k1 = data["k1"]
        instance.b = data["b"]
        instance.bm25 = BM25Okapi(instance.tokenized_corpus, k1=instance.k1, b=instance.b)
        return instance
