"""Retrieval systems package: BM25, Dense, and Hybrid (RRF).

DenseRetriever is NOT imported at package load time because it lazily imports
faiss and sentence_transformers, whose loading is deferred until first use.
Import DenseRetriever directly when needed:

    from src.retrieval.dense import DenseRetriever, DEFAULT_MODEL_ID, BGE_QUERY_PREFIX
"""

from .base import BaseRetriever, ScoredDoc
from .bm25 import BM25Retriever, tokenize_bm25
from .hybrid import HybridRetriever, reciprocal_rank_fusion

__all__ = [
    # Base interface
    "BaseRetriever",
    "ScoredDoc",
    # Sparse retrieval
    "BM25Retriever",
    "tokenize_bm25",
    # Hybrid retrieval
    "HybridRetriever",
    "reciprocal_rank_fusion",
    # Dense retrieval — import explicitly from src.retrieval.dense to avoid eager FAISS load
    # "DenseRetriever",
    # "DEFAULT_MODEL_ID",
    # "BGE_QUERY_PREFIX",
]
