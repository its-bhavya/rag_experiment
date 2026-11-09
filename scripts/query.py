"""Single-query retrieval CLI.

Retrieves top-K documents from one retrieval method for a given query.

Usage:
    python scripts/query.py --query "TEXT" --method bm25 [--k 10]
    python scripts/query.py --query "TEXT" --method dense [--k 10]
    python scripts/query.py --query "TEXT" --method hybrid [--k 10]

Prerequisites:
    Build indexes first:
        python src/experiments/build_indexes.py

Options:
    --query     Query text string (required)
    --method    Retrieval method: bm25 | dense | hybrid (default: bm25)
    --k         Number of results to retrieve (default: 10)
    --corpus    Path to corpus JSONL for displaying titles (optional, default: data/raw/scifact/corpus.jsonl)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets.scifact import parse_corpus
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever

BM25_INDEX_PATH = _ROOT / "results" / "raw" / "bm25_index.pkl"
DENSE_INDEX_DIR = _ROOT / "results" / "raw" / "dense_index"
CORPUS_PATH = _ROOT / "data" / "raw" / "scifact" / "corpus.jsonl"


def load_bm25() -> BM25Retriever:
    if not BM25_INDEX_PATH.exists():
        print(f"[ERROR] BM25 index not found at {BM25_INDEX_PATH}")
        print("        Run: python src/experiments/build_indexes.py")
        sys.exit(1)
    return BM25Retriever.load(BM25_INDEX_PATH)


def load_dense():
    from src.retrieval.dense import DEFAULT_MODEL_ID, DenseRetriever
    if not DENSE_INDEX_DIR.exists():
        print(f"[ERROR] Dense index not found at {DENSE_INDEX_DIR}")
        print("        Run: python src/experiments/build_indexes.py")
        sys.exit(1)
    return DenseRetriever.load(DENSE_INDEX_DIR, device="cpu")


def load_corpus_titles(corpus_path: Path) -> dict[str, str]:
    """Load doc_id -> title mapping for display."""
    if not corpus_path.exists():
        return {}
    corpus = parse_corpus(corpus_path)
    return {doc_id: doc.title for doc_id, doc in corpus.items()}


def print_results(results, titles: dict[str, str], method: str, query: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"Method : {method.upper()}")
    print(f"Query  : {query}")
    print(f"{'=' * 70}")
    print(f"{'Rank':<6} {'Score':>8}  {'Doc ID':<12}  Title")
    print(f"{'-' * 70}")
    for doc in results:
        title = titles.get(doc.doc_id, "(title not available)")
        # Truncate long titles
        title_disp = title[:50] + "..." if len(title) > 50 else title
        print(f"{doc.rank:<6} {doc.score:>8.4f}  {doc.doc_id:<12}  {title_disp}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single query against a retrieval index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--query", required=True, help="Query text string.")
    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "hybrid"],
        default="bm25",
        help="Retrieval method (default: bm25)",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_PATH,
        help="Corpus JSONL path for title display",
    )
    args = parser.parse_args()

    if args.k < 1:
        print("[ERROR] --k must be a positive integer")
        sys.exit(1)

    titles = load_corpus_titles(args.corpus)

    if args.method == "bm25":
        retriever = load_bm25()
    elif args.method == "dense":
        retriever = load_dense()
    elif args.method == "hybrid":
        bm25 = load_bm25()
        dense = load_dense()
        retriever = HybridRetriever(bm25, dense, c=60, candidate_pool_size=max(100, args.k * 10))
    else:
        print(f"[ERROR] Unknown method: {args.method}")
        sys.exit(1)

    results = retriever.retrieve(args.query, k=args.k)
    print_results(results, titles, args.method, args.query)


if __name__ == "__main__":
    main()
