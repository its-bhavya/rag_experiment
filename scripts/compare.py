"""Side-by-side comparison of BM25, Dense, and Hybrid retrieval for a single query.

Usage:
    python scripts/compare.py --query "TEXT" [--k 10]

Prerequisites:
    Build indexes first:
        python src/experiments/build_indexes.py

Output: tabulated top-K results from each method with doc_id, score, rank, and title.
"""

from __future__ import annotations

import argparse
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
    from src.retrieval.dense import DenseRetriever
    if not DENSE_INDEX_DIR.exists():
        print(f"[ERROR] Dense index not found at {DENSE_INDEX_DIR}")
        print("        Run: python src/experiments/build_indexes.py")
        sys.exit(1)
    return DenseRetriever.load(DENSE_INDEX_DIR, device="cpu")


def load_corpus_titles(corpus_path: Path) -> dict[str, str]:
    if not corpus_path.exists():
        return {}
    corpus = parse_corpus(corpus_path)
    return {doc_id: doc.title for doc_id, doc in corpus.items()}


def print_method_results(method_name: str, results, titles: dict[str, str]) -> None:
    print(f"\n  [{method_name.upper()}]")
    print(f"  {'Rank':<5} {'Score':>8}  {'Doc ID':<12}  Title")
    print(f"  {'-' * 65}")
    for doc in results:
        title = titles.get(doc.doc_id, "(n/a)")
        title_disp = title[:48] + "..." if len(title) > 48 else title
        print(f"  {doc.rank:<5} {doc.score:>8.4f}  {doc.doc_id:<12}  {title_disp}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BM25, Dense, and Hybrid retrieval side-by-side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--query", required=True, help="Query text string.")
    parser.add_argument("--k", type=int, default=10, help="Number of results per method (default: 10)")
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

    print(f"\n{'=' * 72}")
    print(f"  RETRIEVAL COMPARISON")
    print(f"  Query : {args.query}")
    print(f"  Top-K : {args.k}")
    print(f"{'=' * 72}")

    titles = load_corpus_titles(args.corpus)

    bm25 = load_bm25()
    dense = load_dense()
    hybrid = HybridRetriever(bm25, dense, c=60, candidate_pool_size=max(100, args.k * 10))

    for method_name, retriever in [("bm25", bm25), ("dense", dense), ("hybrid", hybrid)]:
        results = retriever.retrieve(args.query, k=args.k)
        print_method_results(method_name, results, titles)

    print()


if __name__ == "__main__":
    main()
