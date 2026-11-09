"""Dense retrieval implementation using Sentence Transformers and FAISS.

FAISS and SentenceTransformers are imported lazily at first use, not at module load time.
This allows the rest of the retrieval package (BM25, RRF, Hybrid) to be imported and
tested in environments where the FAISS DLL is unavailable (e.g. sandboxed CI runners).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.datasets.scifact import SciFactDocument
from src.retrieval.base import BaseRetriever, ScoredDoc

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "BAAI/bge-base-en-v1.5"
# Official BGE query instruction prefix for retrieval tasks (from model card)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _import_faiss():
    """Lazily import FAISS; raises ImportError with helpful message on failure."""
    try:
        import faiss  # noqa: PLC0415
        return faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required for DenseRetriever. "
            "Install it with: pip install faiss-cpu"
        ) from exc


def _import_sentence_transformers():
    """Lazily import SentenceTransformer; raises ImportError with helpful message."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        return SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for DenseRetriever. "
            "Install it with: pip install sentence-transformers"
        ) from exc


class DenseRetriever(BaseRetriever):
    """Dense retriever using Sentence Transformers and FAISS FlatIP index.

    Index Details:
        - Model: BAAI/bge-base-en-v1.5 (768-dimensional embeddings)
        - Document format: "{title}\\n{text}".strip()
        - Query format: BGE_QUERY_PREFIX + query.strip()
        - Embeddings L2-normalized; FAISS IndexFlatIP (inner product == cosine similarity)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        query_instruction: str = BGE_QUERY_PREFIX,
        device: str = "cpu",
        batch_size: int = 32,
        max_seq_length: int = 256,
    ) -> None:
        """Initialize dense retriever and load the SentenceTransformer model.

        Args:
            model_id: HuggingFace model ID for SentenceTransformer.
            query_instruction: Task instruction string prefixed to query text at retrieval time.
            device: Compute device: 'cpu' or 'cuda'.
            batch_size: Batch size used for document encoding.
            max_seq_length: Maximum sequence length in tokens (default 256 for fast CPU inference).
        """
        SentenceTransformer = _import_sentence_transformers()

        self.model_id = model_id
        self.query_instruction = query_instruction
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        logger.info(f"Loading SentenceTransformer: {self.model_id} on {self.device}...")
        try:
            self.model = SentenceTransformer(
                self.model_id,
                device=self.device,
                local_files_only=True,
            )
        except Exception:
            self.model = SentenceTransformer(self.model_id, device=self.device)

        if self.max_seq_length:
            self.model.max_seq_length = self.max_seq_length

        self.dimension: int = self.model.get_embedding_dimension()

        self.doc_ids: List[str] = []
        self.index = None  # faiss.IndexFlatIP, set by build_index / load
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}

    def build_index(
        self,
        doc_ids: Sequence[str],
        doc_texts: Sequence[str],
        save_dir: Optional[Path | str] = None,
    ) -> None:
        """Encode documents, build FAISS IndexFlatIP index, and optionally cache artifacts.

        Document Format:
            Each document is encoded as-is (caller is responsible for pre-formatting as
            "{title}\\n{text}".strip()). Embeddings are L2-normalized so inner product
            equals cosine similarity.

        Args:
            doc_ids: Sequence of document IDs (must match doc_texts in length).
            doc_texts: Sequence of pre-formatted document text strings.
            save_dir: If provided, cache FAISS index, embeddings, doc_ids, and metadata here.
        """
        faiss = _import_faiss()

        if len(doc_ids) != len(doc_texts):
            raise ValueError(
                f"Mismatch: len(doc_ids)={len(doc_ids)} vs len(doc_texts)={len(doc_texts)}"
            )

        self.doc_ids = list(doc_ids)
        logger.info(
            f"Encoding {len(doc_texts)} documents with {self.model_id} "
            f"(batch_size={self.batch_size})..."
        )

        # Encode and L2-normalize embeddings (cosine sim = inner product on unit vectors)
        embeddings = self.model.encode(
            list(doc_texts),
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Build FAISS IndexFlatIP (exact search, inner product metric)
        logger.info(f"Building FAISS IndexFlatIP with dimension {self.dimension}...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)

        self.metadata = {
            "model_id": self.model_id,
            "dimension": int(self.dimension),
            "max_seq_length": int(self.max_seq_length),
            "normalized": True,
            "metric": "cosine_inner_product",
            "faiss_index_type": "IndexFlatIP",
            "query_instruction": self.query_instruction,
            "document_format": "title\\ntext",
            "num_documents": len(self.doc_ids),
            "device": self.device,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if save_dir:
            self.save(save_dir)

    @classmethod
    def from_corpus(
        cls,
        corpus: Mapping[str, "SciFactDocument | Dict[str, Any]"],
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        batch_size: int = 32,
        max_seq_length: int = 256,
        save_dir: Optional[Path | str] = None,
    ) -> "DenseRetriever":
        """Factory method to build a dense retriever directly from a corpus dictionary."""
        retriever = cls(
            model_id=model_id,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        )
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

        retriever.build_index(doc_ids=doc_ids, doc_texts=doc_texts, save_dir=save_dir)
        return retriever

    def retrieve(self, query: str, k: int = 10) -> List[ScoredDoc]:
        """Retrieve top-K ranked documents using cosine similarity via FAISS.

        Args:
            query: Query text string (instruction prefix is added automatically).
            k: Number of documents to retrieve.

        Returns:
            List of ScoredDoc ordered by descending cosine similarity with 1-indexed ranks.
        """
        if self.index is None or not self.doc_ids:
            raise RuntimeError("Dense index has not been built or loaded. Call build_index() first.")
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got k={k}")

        # Prefix query with model-specific instruction
        formatted_query = f"{self.query_instruction}{query.strip()}"

        # Encode and normalize query vector to unit length
        q_emb = self.model.encode(
            [formatted_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)

        # Search FAISS index for top-k nearest neighbours by inner product
        num_candidates = min(k, len(self.doc_ids))
        scores, indices = self.index.search(q_emb, num_candidates)

        ranked_docs: List[ScoredDoc] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            ranked_docs.append(
                ScoredDoc(
                    doc_id=self.doc_ids[idx],
                    score=float(score),
                    rank=rank,
                )
            )

        return ranked_docs

    def save(self, save_dir: Path | str) -> None:
        """Cache FAISS index, embeddings, document IDs, and metadata to disk."""
        faiss = _import_faiss()
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(out_dir / "index.faiss"))

        if self.embeddings is not None:
            np.save(out_dir / "embeddings.npy", self.embeddings)

        with open(out_dir / "doc_ids.json", "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f)

        with open(out_dir / "index_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved Dense index and artifacts to {out_dir}")

    @classmethod
    def load(
        cls,
        save_dir: Path | str,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
    ) -> "DenseRetriever":
        """Load cached Dense retriever from disk.

        Args:
            save_dir: Directory previously written by save().
            model_id: Fallback model ID if not stored in metadata.
            device: Compute device.

        Returns:
            Fully initialized DenseRetriever ready for retrieve() calls.
        """
        faiss = _import_faiss()
        in_dir = Path(save_dir)
        index_file = in_dir / "index.faiss"
        doc_ids_file = in_dir / "doc_ids.json"
        meta_file = in_dir / "index_metadata.json"

        if not index_file.exists() or not doc_ids_file.exists():
            raise FileNotFoundError(f"Missing dense index artifacts in {in_dir}")

        meta: Dict[str, Any] = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        retriever = cls(
            model_id=meta.get("model_id", model_id),
            query_instruction=meta.get("query_instruction", BGE_QUERY_PREFIX),
            device=device,
        )

        with open(doc_ids_file, "r", encoding="utf-8") as f:
            retriever.doc_ids = json.load(f)

        retriever.index = faiss.read_index(str(index_file))

        emb_file = in_dir / "embeddings.npy"
        if emb_file.exists():
            retriever.embeddings = np.load(emb_file)

        retriever.metadata = meta
        logger.info(f"Loaded Dense index with {len(retriever.doc_ids)} docs from {in_dir}")
        return retriever
