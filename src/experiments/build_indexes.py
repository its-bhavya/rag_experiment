"""Build and cache BM25 and Dense FAISS indexes for the SciFact corpus.

Usage:
    python src/experiments/build_indexes.py [--model-id BAAI/bge-base-en-v1.5] [--device cpu]

Outputs:
    results/raw/bm25_index.pkl          — Serialized BM25 index
    results/raw/dense_index/            — FAISS index, embeddings, doc_ids, metadata
    results/raw/index_build_log.json    — Timing and configuration record
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Project root on sys.path so `from src.xxx import ...` works when run directly
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets.scifact import parse_corpus
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DEFAULT_MODEL_ID, DenseRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_indexes")

CORPUS_PATH = _ROOT / "data" / "raw" / "scifact" / "corpus.jsonl"
BM25_INDEX_PATH = _ROOT / "results" / "raw" / "bm25_index.pkl"
DENSE_INDEX_DIR = _ROOT / "results" / "raw" / "dense_index"
LOG_PATH = _ROOT / "results" / "raw" / "index_build_log.json"


def build_bm25(doc_ids: list[str], doc_texts: list[str]) -> dict:
    """Build and save BM25 index. Returns timing info."""
    logger.info(f"Building BM25 index over {len(doc_ids)} documents...")
    t0 = time.perf_counter()
    retriever = BM25Retriever(doc_ids=doc_ids, doc_texts=doc_texts)
    retriever.save(BM25_INDEX_PATH)
    elapsed = time.perf_counter() - t0
    logger.info(f"BM25 index saved to {BM25_INDEX_PATH} in {elapsed:.1f}s")
    return {
        "index_type": "BM25Okapi",
        "k1": retriever.k1,
        "b": retriever.b,
        "num_documents": len(doc_ids),
        "output_path": str(BM25_INDEX_PATH),
        "elapsed_seconds": round(elapsed, 2),
    }


def build_dense(
    doc_ids: list[str],
    doc_texts: list[str],
    model_id: str,
    device: str,
    max_seq_length: int = 256,
    batch_size: int = 32,
) -> dict:
    """Build and save Dense FAISS index. Returns timing info."""
    logger.info(
        f"Building Dense index with {model_id} on {device} (max_seq_length={max_seq_length}, batch_size={batch_size})..."
    )
    t0 = time.perf_counter()
    retriever = DenseRetriever(
        model_id=model_id,
        device=device,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )
    retriever.build_index(doc_ids=doc_ids, doc_texts=doc_texts, save_dir=DENSE_INDEX_DIR)
    elapsed = time.perf_counter() - t0
    logger.info(f"Dense index saved to {DENSE_INDEX_DIR} in {elapsed:.1f}s")
    return {
        "index_type": "FAISS_IndexFlatIP",
        "model_id": model_id,
        "dimension": retriever.dimension,
        "max_seq_length": max_seq_length,
        "normalized": True,
        "num_documents": len(doc_ids),
        "output_dir": str(DENSE_INDEX_DIR),
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 and Dense retrieval indexes.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID for dense retrieval (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for dense model encoding (default: cpu)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Max sequence length in tokens (default: 256 for fast CPU inference)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for dense encoding (default: 32)",
    )
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Skip building the dense index (useful for quick BM25-only tests)",
    )
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        logger.error(f"Corpus not found at {CORPUS_PATH}. Run dataset download first.")
        sys.exit(1)

    logger.info(f"Loading corpus from {CORPUS_PATH}...")
    corpus = parse_corpus(CORPUS_PATH)
    logger.info(f"Corpus loaded: {len(corpus)} documents.")

    # Format: title + newline + text (consistent with retrieval format in dense.py)
    doc_ids: list[str] = []
    doc_texts: list[str] = []
    for doc_id, doc in corpus.items():
        doc_ids.append(str(doc_id))
        doc_texts.append(doc.full_text)

    build_log: dict = {
        "corpus_path": str(CORPUS_PATH),
        "num_documents": len(doc_ids),
        "indexes": {},
    }

    # --- BM25 ---
    bm25_info = build_bm25(doc_ids, doc_texts)
    build_log["indexes"]["bm25"] = bm25_info

    # --- Dense ---
    if not args.skip_dense:
        dense_info = build_dense(
            doc_ids,
            doc_texts,
            args.model_id,
            args.device,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
        )
        build_log["indexes"]["dense"] = dense_info
    else:
        logger.info("Skipping dense index (--skip-dense flag set).")

    # Save build log
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(build_log, f, indent=2)
    logger.info(f"Build log saved to {LOG_PATH}")
    logger.info("Index building complete.")


if __name__ == "__main__":
    main()
