"""Reproducible experiment runner for sparse, dense, and hybrid retrieval.

Executes BM25, Dense (BAAI/bge-base-en-v1.5 + FAISS), and Hybrid (RRF, c=60)
retrieval over identical queries and corpus documents. Computes conventional
document metrics and evidence completeness metrics.

Outputs:
    results/raw/scifact_runs.jsonl
    results/metrics/scifact_per_query.csv
    results/metrics/scifact_summary.csv
    results/metrics/scifact_run_metadata.json

Usage:
    python -m src.experiments.run_experiments --dataset scifact
    python -m src.experiments.run_experiments --dataset scifact --overwrite
    python -m src.experiments.run_experiments --dataset scifact --methods bm25 --k-values 3 5 7 9
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets.evidence_mapper import load_verified_evidence_ground_truth
from src.datasets.scifact import (
    SciFactDataset,
    SciFactDocument,
    SciFactQuery,
    compute_sha256,
    load_scifact_dataset,
    parse_corpus,
)
from src.evaluation.metrics import (
    EvidenceItem,
    QueryResult,
    RetrievedPassage,
    evaluate_query,
    get_covered_evidence,
    map_passages_to_documents,
)
from src.retrieval.base import BaseRetriever, ScoredDoc
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import DEFAULT_RRF_C, HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiments")

# Fixed default paths
RAW_RUNS_PATH = _ROOT / "results" / "raw" / "scifact_runs.jsonl"
PER_QUERY_CSV_PATH = _ROOT / "results" / "metrics" / "scifact_per_query.csv"
SUMMARY_CSV_PATH = _ROOT / "results" / "metrics" / "scifact_summary.csv"
RUN_METADATA_PATH = _ROOT / "results" / "metrics" / "scifact_run_metadata.json"

BM25_INDEX_PATH = _ROOT / "results" / "raw" / "bm25_index.pkl"
DENSE_INDEX_DIR = _ROOT / "results" / "raw" / "dense_index"
EVIDENCE_JSONL_PATH = _ROOT / "data" / "processed" / "scifact_evidence.jsonl"
EVIDENCE_REPORT_PATH = _ROOT / "data" / "processed" / "evidence_mapping_report.json"
CORPUS_PATH = _ROOT / "data" / "raw" / "scifact" / "corpus.jsonl"


def set_fixed_seeds(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_environment_info() -> Dict[str, Any]:
    """Capture runtime environment details for scientific reproducibility."""
    import importlib.metadata

    env: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }

    # Library versions via package metadata (avoids importing heavy C extensions / DLLs)
    lib_versions: Dict[str, str] = {}
    package_map = {
        "numpy": "numpy",
        "pandas": "pandas",
        "rank_bm25": "rank-bm25",
        "faiss": "faiss-cpu",
        "torch": "torch",
        "sentence_transformers": "sentence-transformers",
    }
    for display_name, pkg_name in package_map.items():
        try:
            lib_versions[display_name] = importlib.metadata.version(pkg_name)
        except Exception:
            lib_versions[display_name] = "not_available"

    env["library_versions"] = lib_versions
    return env


def compute_dataset_fingerprint(data_dir: Path) -> Dict[str, str]:
    """Compute SHA256 checksums of core dataset files."""
    fingerprint: Dict[str, str] = {}
    files_to_hash = [
        data_dir / "corpus.jsonl",
        data_dir / "queries.jsonl",
        data_dir / "qrels" / "test.tsv",
        EVIDENCE_JSONL_PATH,
    ]
    for p in files_to_hash:
        if p.exists():
            fingerprint[p.name] = compute_sha256(p)
    return fingerprint


def get_bm25_retriever(
    corpus: Mapping[str, SciFactDocument],
    save_path: Path = BM25_INDEX_PATH,
) -> BM25Retriever:
    """Load cached BM25 index if available, or build and save."""
    if save_path.exists():
        logger.info(f"Loading cached BM25 index from {save_path}...")
        return BM25Retriever.load(save_path)

    logger.info(f"Building BM25 index over {len(corpus)} documents...")
    retriever = BM25Retriever.from_corpus(corpus)
    retriever.save(save_path)
    return retriever


def get_dense_retriever(
    corpus: Mapping[str, SciFactDocument],
    device: str = "cpu",
    max_seq_length: int = 256,
    batch_size: int = 32,
    save_dir: Path = DENSE_INDEX_DIR,
) -> Any:
    """Load cached Dense retriever if available, or build and save."""
    from src.retrieval.dense import DEFAULT_MODEL_ID, DenseRetriever

    if (save_dir / "index.faiss").exists() and (save_dir / "doc_ids.json").exists():
        logger.info(f"Loading cached Dense index from {save_dir}...")
        return DenseRetriever.load(save_dir, device=device)

    logger.info(
        f"Dense index not found at {save_dir}. Building from corpus ({len(corpus)} docs, max_seq_length={max_seq_length})..."
    )
    doc_ids = [str(doc_id) for doc_id in corpus.keys()]
    doc_texts = [doc.full_text for doc in corpus.values()]
    retriever = DenseRetriever(
        model_id=DEFAULT_MODEL_ID,
        device=device,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )
    retriever.build_index(doc_ids=doc_ids, doc_texts=doc_texts, save_dir=save_dir)
    return retriever


def load_completed_run_keys(runs_path: Path) -> Set[Tuple[str, str, int]]:
    """Inspect existing runs file to find completed (query_id, method, K) runs for resumption."""
    completed: Set[Tuple[str, str, int]] = set()
    if not runs_path.exists():
        return completed

    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed.add((str(record["query_id"]), str(record["method"]), int(record["K"])))
            except Exception:
                continue
    return completed


def run_experiment_pipeline(
    dataset_name: str = "scifact",
    k_values: Sequence[int] = (3, 5, 7, 9),
    methods: Sequence[str] = ("bm25", "dense", "hybrid"),
    seed: int = 42,
    device: str = "cpu",
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full retrieval and evaluation experiment suite.

    Args:
        dataset_name: Name of the dataset (currently 'scifact').
        k_values: Cutoff ranks to evaluate (e.g. [5, 10]).
        methods: Retrieval methods to run (subset of ['bm25', 'dense', 'hybrid']).
        seed: Fixed random seed.
        device: 'cpu' or 'cuda'.
        overwrite: If True, warns and clears existing results; if False, safely resumes.
        limit: Optional maximum number of queries to evaluate (useful for testing).

    Returns:
        Tuple of (per_query_dataframe, summary_dataframe).
    """
    set_fixed_seeds(seed)
    max_k = max(k_values)

    logger.info(f"=== Starting Experiment Runner: dataset={dataset_name}, seed={seed} ===")
    logger.info(f"Methods: {list(methods)} | K values: {list(k_values)} | Device: {device}")

    # 1. Load Dataset
    data_dir = _ROOT / "data" / "raw" / "scifact"
    dataset: SciFactDataset = load_scifact_dataset(data_dir=data_dir, split="test")
    corpus = dataset.corpus
    test_queries = dataset.queries
    test_qrels = dataset.qrels

    # Select queries present in test qrels (300 test queries)
    eval_qids = [qid for qid in test_queries.keys() if qid in test_qrels]
    if limit is not None and limit > 0:
        logger.info(f"Limiting evaluation to first {limit} test queries.")
        eval_qids = eval_qids[:limit]

    logger.info(f"Evaluating {len(eval_qids)} queries across {len(corpus)} corpus documents.")

    # 2. Load Verified Evidence Ground Truth
    evidence_by_query: Dict[str, List[EvidenceItem]] = {}
    eligible_evidence_qids: Set[str] = set()
    if EVIDENCE_JSONL_PATH.exists() and EVIDENCE_REPORT_PATH.exists():
        evidence_by_query, eligible_evidence_qids = load_verified_evidence_ground_truth(
            evidence_jsonl=EVIDENCE_JSONL_PATH,
            report_json=EVIDENCE_REPORT_PATH,
        )
        logger.info(
            f"Loaded evidence ground truth: {len(eligible_evidence_qids)} eligible complete queries."
        )
    else:
        logger.warning(
            f"Evidence artifacts not found at {EVIDENCE_JSONL_PATH}. Evidence metrics will be skipped."
        )

    # 3. Initialize Retrieval Systems
    retrievers: Dict[str, BaseRetriever] = {}
    if "bm25" in methods or "hybrid" in methods:
        retrievers["bm25"] = get_bm25_retriever(corpus)

    if "dense" in methods or "hybrid" in methods:
        retrievers["dense"] = get_dense_retriever(corpus, device=device)

    if "hybrid" in methods:
        retrievers["hybrid"] = HybridRetriever(
            sparse_retriever=retrievers["bm25"],
            dense_retriever=retrievers["dense"],
            c=DEFAULT_RRF_C,
            candidate_pool_size=max(100, max_k * 10),
        )

    # 4. Handle Existing Results & Resumption
    RAW_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PER_QUERY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    completed_runs: Set[Tuple[str, str, int]] = set()
    if RAW_RUNS_PATH.exists():
        if overwrite:
            logger.warning(f"Overwriting existing results at {RAW_RUNS_PATH} (--overwrite specified).")
            RAW_RUNS_PATH.unlink()
        else:
            completed_runs = load_completed_run_keys(RAW_RUNS_PATH)
            logger.info(
                f"Existing results found: {len(completed_runs)} (query, method, K) runs already completed. Resuming..."
            )

    # 5. Run Per-Query Retrieval and Evaluation
    new_records_count = 0
    runs_file_handle = open(RAW_RUNS_PATH, "a", encoding="utf-8")

    try:
        for method_name in methods:
            retriever = retrievers[method_name]
            logger.info(f"Running retrieval method: {method_name.upper()}...")

            for q_idx, qid in enumerate(eval_qids, start=1):
                # If all requested K values are already completed for this query and method, skip retrieval
                if all((str(qid), str(method_name), int(k)) in completed_runs for k in k_values):
                    continue

                query_obj = test_queries[qid]
                query_text = query_obj.text
                relevant_docs = list(test_qrels.get(qid, {}).keys())
                gold_ev = evidence_by_query.get(qid, None)

                # Measure retrieval latency for the maximum required K
                t0 = time.perf_counter()
                retrieved_docs: List[ScoredDoc] = retriever.retrieve(query_text, k=max_k)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                for k in k_values:
                    run_key = (str(qid), str(method_name), int(k))
                    if run_key in completed_runs:
                        continue

                    # Slice top-K candidates
                    top_k_candidates = retrieved_docs[:k]
                    retrieved_doc_ids = [d.doc_id for d in top_k_candidates]
                    ranked_scores = [d.score for d in top_k_candidates]

                    # Document metrics
                    eval_res = evaluate_query(
                        query_id=qid,
                        retrieved_docs=retrieved_doc_ids,
                        relevant_docs=relevant_docs,
                        gold_evidence=gold_ev if qid in eligible_evidence_qids else None,
                        cutoffs=[k],
                        empty_relevant_handling="raise",
                    )
                    metrics = eval_res.metrics

                    # Identify covered evidence units
                    covered_units: List[Dict[str, Any]] = []
                    if gold_ev and qid in eligible_evidence_qids:
                        covered_set = get_covered_evidence(retrieved_doc_ids, gold_ev, k=k)
                        covered_units = [
                            {"doc_id": e.doc_id, "evidence_id": e.evidence_id, "text": e.text}
                            for e in covered_set
                        ]

                    run_record: Dict[str, Any] = {
                        "dataset": dataset_name,
                        "query_id": str(qid),
                        "query_text": query_text,
                        "method": method_name,
                        "K": int(k),
                        "retrieved_document_ids": retrieved_doc_ids,
                        "retrieved_passage_ids": None,
                        "ranked_scores": ranked_scores,
                        "latency_ms": round(elapsed_ms, 3),
                        "relevant_document_ids": relevant_docs,
                        "evidence_units_retrieved": covered_units,
                        "query_level_metrics": metrics,
                    }

                    runs_file_handle.write(json.dumps(run_record, ensure_ascii=False) + "\n")
                    runs_file_handle.flush()
                    completed_runs.add(run_key)
                    new_records_count += 1

                if q_idx % 50 == 0 or q_idx == len(eval_qids):
                    logger.info(
                        f"  [{method_name.upper()}] Processed {q_idx}/{len(eval_qids)} queries..."
                    )

    finally:
        runs_file_handle.close()

    logger.info(f"Retrieval complete. Added {new_records_count} new run records.")

    # 6. Aggregate All Accumulated Runs into CSV Tables
    logger.info("Aggregating runs into per-query and summary CSV tables...")
    all_records: List[Dict[str, Any]] = []
    with open(RAW_RUNS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    per_query_rows: List[Dict[str, Any]] = []
    for r in all_records:
        k = r["K"]
        m = r.get("query_level_metrics", {})
        num_evidence_annotated = (
            len(evidence_by_query[r["query_id"]])
            if r["query_id"] in eligible_evidence_qids
            else None
        )
        num_evidence_retrieved = (
            len(r["evidence_units_retrieved"])
            if r["query_id"] in eligible_evidence_qids
            else None
        )

        per_query_rows.append(
            {
                "dataset": r["dataset"],
                "query_id": r["query_id"],
                "method": r["method"],
                "K": k,
                "latency_ms": r["latency_ms"],
                "num_retrieved": len(r["retrieved_document_ids"]),
                "num_relevant": len(r["relevant_document_ids"]),
                "num_relevant_retrieved": sum(
                    1 for d in r["retrieved_document_ids"] if d in set(r["relevant_document_ids"])
                ),
                "num_evidence_annotated": num_evidence_annotated,
                "num_evidence_retrieved": num_evidence_retrieved,
                "Recall@K": m.get(f"Recall@{k}"),
                "Precision@K": m.get(f"Precision@{k}"),
                "MRR": m.get("MRR"),
                "EvidenceRecall@K": m.get(f"EvidenceRecall@{k}"),
                "EvidenceCompleteness@K": m.get(f"EvidenceCompleteness@{k}"),
            }
        )

    df_per_query = pd.DataFrame(per_query_rows)
    df_per_query.to_csv(PER_QUERY_CSV_PATH, index=False)
    logger.info(f"Saved per-query metrics to {PER_QUERY_CSV_PATH} ({len(df_per_query)} rows).")

    # 7. Generate Summary Table
    # Summary columns: dataset, method, K, query_count, Recall@K, Precision@K, MRR, Evidence Recall@K, Evidence Completeness@K, mean latency, median latency
    all_methods_in_runs = sorted(df_per_query["method"].unique())
    all_k_in_runs = sorted(df_per_query["K"].unique())
    summary_rows: List[Dict[str, Any]] = []
    for method in all_methods_in_runs:
        for k in all_k_in_runs:
            subset = df_per_query[(df_per_query["method"] == method) & (df_per_query["K"] == k)]
            if subset.empty:
                continue

            query_count = len(subset)
            mean_recall = float(subset["Recall@K"].mean())
            mean_precision = float(subset["Precision@K"].mean())
            mean_mrr = float(subset["MRR"].mean())

            # Evidence metrics evaluated over queries with complete annotations
            ev_subset = subset[subset["EvidenceRecall@K"].notna()]
            mean_ev_recall = float(ev_subset["EvidenceRecall@K"].mean()) if not ev_subset.empty else 0.0
            mean_ev_completeness = (
                float(ev_subset["EvidenceCompleteness@K"].mean()) if not ev_subset.empty else 0.0
            )

            mean_latency = float(subset["latency_ms"].mean())
            median_latency = float(subset["latency_ms"].median())

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "K": k,
                    "query_count": query_count,
                    "Recall@K": round(mean_recall, 4),
                    "Precision@K": round(mean_precision, 4),
                    "MRR": round(mean_mrr, 4),
                    "Evidence Recall@K": round(mean_ev_recall, 4),
                    "Evidence Completeness@K": round(mean_ev_completeness, 4),
                    "mean latency": round(mean_latency, 3),
                    "median latency": round(median_latency, 3),
                }
            )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(SUMMARY_CSV_PATH, index=False)
    logger.info(f"Saved summary metrics to {SUMMARY_CSV_PATH}.")

    # Print nicely formatted summary table to console
    print("\n" + "=" * 90)
    print("                     RETRIEVAL EVALUATION SUMMARY TABLE")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90 + "\n")

    # 8. Save Run Metadata & Provenance
    metadata: Dict[str, Any] = {
        "dataset": dataset_name,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "k_values": list(k_values),
        "methods": list(methods),
        "environment": get_environment_info(),
        "dataset_fingerprint": compute_dataset_fingerprint(data_dir),
        "retrieval_configuration": {
            "bm25": {"k1": 1.5, "b": 0.75, "tokenizer": "tokenize_bm25_nfkd"},
            "dense": {
                "model_id": "BAAI/bge-base-en-v1.5",
                "dimension": 768,
                "faiss_index_type": "IndexFlatIP",
                "normalized": True,
                "query_instruction": "Represent this sentence for searching relevant passages: ",
            },
            "hybrid": {"rrf_c": DEFAULT_RRF_C, "candidate_pool_size": 100},
        },
        "query_counts": {
            "total_evaluated_queries": len(eval_qids),
            "evidence_evaluated_queries": len(
                [qid for qid in eval_qids if qid in eligible_evidence_qids]
            ),
        },
    }

    with open(RUN_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved run metadata to {RUN_METADATA_PATH}")

    return df_per_query, df_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible IR and evidence completeness experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="scifact",
        choices=["scifact"],
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[3, 5, 7, 9],
        help="Cutoff ranks K.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bm25", "dense", "hybrid"],
        choices=["bm25", "dense", "hybrid"],
        help="Retrieval methods to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed random seed.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for dense embedding computation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previous results instead of safely resuming.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries (for testing/debugging).",
    )
    args = parser.parse_args()

    run_experiment_pipeline(
        dataset_name=args.dataset,
        k_values=args.k_values,
        methods=args.methods,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
