"""HotpotQA distractor-setting experiment runner.

Loads the processed HotpotQA data, builds **per-query** retrieval indices over
each question's private 10-paragraph candidate pool, runs BM25, Dense, and
Hybrid retrieval at K=5 and K=10, computes all retrieval and evidence metrics,
and saves raw runs plus per-query and summary CSVs.

LIMITATION:
    Each question's candidate pool contains only the 10 distractor paragraphs
    supplied by HotpotQA.  This is a closed distractor experiment, NOT
    full-Wikipedia retrieval.

Usage::

    python -m src.experiments.run_hotpotqa
    python -m src.experiments.run_hotpotqa --methods bm25 dense hybrid --k-values 5 10
    python -m src.experiments.run_hotpotqa --sample-size 500 --seed 12345
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path before local imports
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    EvidenceItem,
    RetrievedPassage,
    evaluate_query,
)
from src.retrieval.base import BaseRetriever, ScoredDoc  # noqa: E402
from src.retrieval.bm25 import BM25Retriever, tokenize_bm25  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_hotpotqa")

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = _ROOT / "data" / "processed"
RESULTS_RAW_DIR = _ROOT / "results" / "raw"
RESULTS_METRICS_DIR = _ROOT / "results" / "metrics"

RAW_RUNS_PATH = RESULTS_RAW_DIR / "hotpotqa_runs.jsonl"
PER_QUERY_CSV_PATH = RESULTS_METRICS_DIR / "hotpotqa_per_query.csv"
SUMMARY_CSV_PATH = RESULTS_METRICS_DIR / "hotpotqa_summary.csv"

# BGE query instruction prefix (must match DenseRetriever)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Per-Query Dense Retriever (numpy brute-force, no FAISS needed for 10 docs)
# ---------------------------------------------------------------------------


class PerQueryDenseRetriever(BaseRetriever):
    """In-memory dense retriever using numpy cosine similarity.

    Designed for small per-query candidate pools (e.g. 10 paragraphs).
    Uses L2-normalised embeddings with dot-product == cosine similarity.
    The SentenceTransformer model is shared across retriever instances to
    avoid repeated loading.

    Parameters
    ----------
    doc_ids:
        List of paragraph doc_ids.
    doc_texts:
        Corresponding full-text strings (pre-formatted).
    model:
        A loaded ``SentenceTransformer`` instance (shared).
    query_instruction:
        Instruction prefix added to each query before encoding.
    """

    def __init__(
        self,
        doc_ids: List[str],
        doc_texts: List[str],
        model: Any,
        query_instruction: str = BGE_QUERY_PREFIX,
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        if len(doc_ids) != len(doc_texts):
            raise ValueError(
                f"doc_ids and doc_texts length mismatch: "
                f"{len(doc_ids)} vs {len(doc_texts)}"
            )
        self.doc_ids = list(doc_ids)
        self.query_instruction = query_instruction
        self.model = model
        self._cached_ranking: Optional[Tuple[str, List[Tuple[str, float]]]] = None

        if embeddings is not None:
            self.embeddings = embeddings
        else:
            # Encode and L2-normalise document embeddings
            raw = self.model.encode(
                doc_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            self.embeddings = np.ascontiguousarray(raw, dtype=np.float32)

    def retrieve(self, query: str, k: int = 10) -> List[ScoredDoc]:
        """Retrieve top-K docs by cosine similarity (brute-force numpy dot)."""
        if k <= 0:
            raise ValueError(f"k must be positive, got k={k}")

        if self._cached_ranking is not None and self._cached_ranking[0] == query:
            ranked_items = self._cached_ranking[1]
        else:
            formatted_query = f"{self.query_instruction}{query.strip()}"
            q_emb = self.model.encode(
                [formatted_query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            q_emb = np.ascontiguousarray(q_emb, dtype=np.float32).squeeze(0)

            # cosine similarity via dot product (both sides are L2-normalised)
            scores = self.embeddings.dot(q_emb).tolist()

            indexed = sorted(
                range(len(self.doc_ids)),
                key=lambda i: (-scores[i], self.doc_ids[i]),
            )
            ranked_items = [(self.doc_ids[i], float(scores[i])) for i in indexed]
            self._cached_ranking = (query, ranked_items)

        num_candidates = min(k, len(ranked_items))
        return [
            ScoredDoc(doc_id=doc_id, score=score, rank=rank)
            for rank, (doc_id, score) in enumerate(ranked_items[:num_candidates], start=1)
        ]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_processed_data(
    processed_dir: Path,
) -> Tuple[
    Dict[str, Any],  # questions: question_id -> question record
    Dict[str, Dict[str, Any]],  # candidates: question_id -> {doc_id -> candidate}
    Dict[str, List[EvidenceItem]],  # evidence: question_id -> [EvidenceItem]
]:
    """Load hotpotqa_questions.jsonl, hotpotqa_candidates.jsonl, hotpotqa_evidence.jsonl."""
    questions: Dict[str, Any] = {}
    q_path = processed_dir / "hotpotqa_questions.jsonl"
    if not q_path.exists():
        raise FileNotFoundError(f"Questions file not found: {q_path}")
    with open(q_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            questions[rec["question_id"]] = rec
    logger.info("Loaded %d questions from %s", len(questions), q_path)

    # candidates: question_id -> {doc_id: candidate_record}
    candidates: Dict[str, Dict[str, Any]] = {qid: {} for qid in questions}
    c_path = processed_dir / "hotpotqa_candidates.jsonl"
    if not c_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {c_path}")
    with open(c_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qid = rec["question_id"]
            if qid in candidates:
                candidates[qid][rec["doc_id"]] = rec
    total_paras = sum(len(v) for v in candidates.values())
    logger.info("Loaded %d candidate paragraphs from %s", total_paras, c_path)

    # evidence items
    evidence: Dict[str, List[EvidenceItem]] = {qid: [] for qid in questions}
    e_path = processed_dir / "hotpotqa_evidence.jsonl"
    if not e_path.exists():
        raise FileNotFoundError(f"Evidence file not found: {e_path}")
    with open(e_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qid = rec["query_id"]
            if qid in evidence:
                evidence[qid].append(
                    EvidenceItem(
                        doc_id=rec["evidence_doc_id"],
                        evidence_id=rec["sentence_idx"],
                        text=rec["evidence_sentence"],
                    )
                )
    total_ev = sum(len(v) for v in evidence.values())
    logger.info("Loaded %d evidence items from %s", total_ev, e_path)

    return questions, candidates, evidence


def _load_completed_runs(raw_runs_path: Path) -> Set[Tuple[str, str, int]]:
    """Return set of (question_id, method, k) tuples already in the runs file."""
    completed: Set[Tuple[str, str, int]] = set()
    if not raw_runs_path.exists():
        return completed
    with open(raw_runs_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                completed.add((rec["query_id"], rec["method"], rec["k"]))
            except (json.JSONDecodeError, KeyError):
                continue
    logger.info("Found %d already-completed runs in %s", len(completed), raw_runs_path)
    return completed


# ---------------------------------------------------------------------------
# Per-query retriever construction
# ---------------------------------------------------------------------------


def _build_bm25(
    doc_ids: List[str],
    doc_texts: List[str],
) -> BM25Retriever:
    return BM25Retriever(doc_ids=doc_ids, doc_texts=doc_texts)


def _build_dense(
    doc_ids: List[str],
    doc_texts: List[str],
    model: Any,
    doc_embeddings_cache: Optional[Dict[str, np.ndarray]] = None,
) -> PerQueryDenseRetriever:
    embs: Optional[np.ndarray] = None
    if doc_embeddings_cache is not None and all(d in doc_embeddings_cache for d in doc_ids):
        embs = np.ascontiguousarray([doc_embeddings_cache[d] for d in doc_ids], dtype=np.float32)
    return PerQueryDenseRetriever(doc_ids=doc_ids, doc_texts=doc_texts, model=model, embeddings=embs)


def _build_hybrid(
    bm25: BM25Retriever,
    dense: PerQueryDenseRetriever,
    candidate_pool_size: int = 10,
) -> HybridRetriever:
    return HybridRetriever(
        sparse_retriever=bm25,
        dense_retriever=dense,
        candidate_pool_size=candidate_pool_size,
    )


# ---------------------------------------------------------------------------
# Sentence Transformer lazy loader (shared model)
# ---------------------------------------------------------------------------

_SHARED_MODEL: Any = None


def _get_model(model_id: str = "BAAI/bge-base-en-v1.5") -> Any:
    """Load (or return cached) SentenceTransformer model."""
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            logger.info("Loading SentenceTransformer: %s ...", model_id)
            try:
                _SHARED_MODEL = SentenceTransformer(
                    model_id, device="cpu", local_files_only=True
                )
            except Exception:
                _SHARED_MODEL = SentenceTransformer(model_id, device="cpu")
            _SHARED_MODEL.max_seq_length = 256
            logger.info("Model loaded successfully.")
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc
    return _SHARED_MODEL


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_hotpotqa_experiment(
    methods: List[str],
    k_values: List[int],
    processed_dir: Path = PROCESSED_DIR,
    raw_runs_path: Path = RAW_RUNS_PATH,
    per_query_csv_path: Path = PER_QUERY_CSV_PATH,
    summary_csv_path: Path = SUMMARY_CSV_PATH,
    model_id: str = "BAAI/bge-base-en-v1.5",
    overwrite: bool = False,
) -> None:
    """Run BM25 / Dense / Hybrid retrieval over the HotpotQA distractor dataset.

    Parameters
    ----------
    methods:
        Retrieval methods to run: any subset of ``["bm25", "dense", "hybrid"]``.
    k_values:
        Rank cutoffs to evaluate.
    processed_dir:
        Directory containing processed hotpotqa files.
    raw_runs_path / per_query_csv_path / summary_csv_path:
        Output file paths.
    model_id:
        HuggingFace model ID for dense retrieval.
    overwrite:
        If True, clears existing run records and recomputes from scratch.
    """
    logger.warning(
        "LIMITATION: This is a closed distractor experiment. "
        "Candidate corpora contain only the 10 paragraphs supplied per question. "
        "This is NOT full-Wikipedia retrieval."
    )

    # Prepare output directories
    raw_runs_path.parent.mkdir(parents=True, exist_ok=True)
    per_query_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite:
        logger.warning("Overwriting existing HotpotQA results (--overwrite specified).")
        if raw_runs_path.exists():
            raw_runs_path.unlink()
        if per_query_csv_path.exists():
            per_query_csv_path.unlink()
        if summary_csv_path.exists():
            summary_csv_path.unlink()

    questions, candidates, evidence = _load_processed_data(processed_dir)
    completed = _load_completed_runs(raw_runs_path)

    # Load model once if dense/hybrid requested
    model: Optional[Any] = None
    if "dense" in methods or "hybrid" in methods:
        model = _get_model(model_id)

    # Pre-encode or load cached candidate paragraph embeddings
    doc_embeddings_cache: Optional[Dict[str, np.ndarray]] = None
    if ("dense" in methods or "hybrid" in methods) and model is not None:
        emb_cache_file = raw_runs_path.parent / "hotpotqa_candidate_embeddings.npy"
        doc_ids_cache_file = raw_runs_path.parent / "hotpotqa_candidate_doc_ids.json"
        if emb_cache_file.exists() and doc_ids_cache_file.exists():
            logger.info("Loading cached candidate embeddings from %s...", emb_cache_file)
            raw_embs = np.load(emb_cache_file)
            with open(doc_ids_cache_file, encoding="utf-8") as f:
                cached_ids = json.load(f)
            doc_embeddings_cache = {did: raw_embs[i] for i, did in enumerate(cached_ids)}
            logger.info("Loaded embeddings for %d candidates.", len(doc_embeddings_cache))
        else:
            logger.info("Pre-encoding all candidate paragraphs in batches...")
            all_candidate_ids: List[str] = []
            all_candidate_texts: List[str] = []
            for q_cand in candidates.values():
                for did, cand_rec in q_cand.items():
                    all_candidate_ids.append(did)
                    all_candidate_texts.append(cand_rec["full_text"])

            raw_embs = model.encode(
                all_candidate_texts,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            raw_embs = np.ascontiguousarray(raw_embs, dtype=np.float32)
            np.save(emb_cache_file, raw_embs)
            with open(doc_ids_cache_file, "w", encoding="utf-8") as f:
                json.dump(all_candidate_ids, f)
            doc_embeddings_cache = {did: raw_embs[i] for i, did in enumerate(all_candidate_ids)}
            logger.info("Cached embeddings for %d candidate paragraphs.", len(doc_embeddings_cache))

    # Collect per-query results for CSV writing
    per_query_rows: List[Dict[str, Any]] = []
    # Accumulate for summary: (method, k) -> list of metric dicts
    from collections import defaultdict

    summary_accum: Dict[Tuple[str, int], List[Dict[str, float]]] = defaultdict(list)
    latency_accum: Dict[Tuple[str, int], List[float]] = defaultdict(list)

    # Silence verbose BM25 build logs during per-query iterations
    logging.getLogger("src.retrieval.bm25").setLevel(logging.WARNING)

    total_q = len(questions)
    for q_num, (qid, q_rec) in enumerate(questions.items(), start=1):
        # Quick skip if all requested runs for this question are already completed
        needed_methods = [
            m for m in methods
            if any((qid, m, k) not in completed for k in k_values)
        ]
        if not needed_methods:
            continue

        q_candidates = candidates[qid]
        q_evidence = evidence[qid]
        q_text: str = q_rec["question"]
        gold_doc_ids: List[str] = q_rec["gold_doc_ids"]

        if not q_candidates:
            logger.warning("Question %s has no candidate paragraphs — skipping.", qid)
            continue

        doc_ids_list = list(q_candidates.keys())
        doc_texts_list = [q_candidates[d]["full_text"] for d in doc_ids_list]

        # Build retrievers (lazy — only when needed)
        bm25_retriever: Optional[BM25Retriever] = None
        dense_retriever: Optional[PerQueryDenseRetriever] = None
        hybrid_retriever: Optional[HybridRetriever] = None

        def _get_bm25() -> BM25Retriever:
            nonlocal bm25_retriever
            if bm25_retriever is None:
                bm25_retriever = _build_bm25(doc_ids_list, doc_texts_list)
            return bm25_retriever

        def _get_dense() -> PerQueryDenseRetriever:
            nonlocal dense_retriever
            if dense_retriever is None:
                dense_retriever = _build_dense(
                    doc_ids_list,
                    doc_texts_list,
                    model,
                    doc_embeddings_cache=doc_embeddings_cache,
                )
            return dense_retriever

        def _get_hybrid() -> HybridRetriever:
            nonlocal hybrid_retriever
            if hybrid_retriever is None:
                hybrid_retriever = _build_hybrid(_get_bm25(), _get_dense(), candidate_pool_size=len(doc_ids_list))
            return hybrid_retriever

        retriever_map = {
            "bm25": _get_bm25,
            "dense": _get_dense,
            "hybrid": _get_hybrid,
        }

        if q_num % 50 == 0 or q_num == 1:
            logger.info("Processing question %d/%d (id=%s)", q_num, total_q, qid)

        for method in needed_methods:
            if method not in retriever_map:
                logger.warning("Unknown method '%s' — skipping.", method)
                continue

            retriever = retriever_map[method]()

            for k in k_values:
                if (qid, method, k) in completed:
                    continue

                # Time retrieval
                t0 = time.perf_counter()
                scored_docs = retriever.retrieve(q_text, k=k)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                # Build RetrievedPassage objects for evidence-aware evaluation
                retrieved_passages: List[RetrievedPassage] = []
                for sd in scored_docs:
                    para = q_candidates.get(sd.doc_id, {})
                    full_text = para.get("full_text", "")
                    retrieved_passages.append(
                        RetrievedPassage(
                            doc_id=sd.doc_id,
                            is_full_document=True,
                            text=full_text,
                            score=sd.score,
                            rank=sd.rank,
                        )
                    )

                # Evaluate
                eval_result = evaluate_query(
                    query_id=qid,
                    retrieved_docs=retrieved_passages,
                    relevant_docs=gold_doc_ids,
                    gold_evidence=q_evidence if q_evidence else None,
                    cutoffs=[k],
                    empty_relevant_handling="zero",
                    empty_evidence_handling="zero",
                )

                # Write raw run record
                run_record: Dict[str, Any] = {
                    "query_id": qid,
                    "method": method,
                    "k": k,
                    "retrieved_doc_ids": [sd.doc_id for sd in scored_docs],
                    "ranked_scores": [sd.score for sd in scored_docs],
                    "relevant_doc_ids": gold_doc_ids,
                    "latency_ms": latency_ms,
                    "metrics": eval_result.metrics,
                }
                with open(raw_runs_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(run_record, ensure_ascii=False) + "\n")

                m = eval_result.metrics
                per_query_rows.append(
                    {
                        "dataset": "hotpotqa",
                        "query_id": qid,
                        "method": method,
                        "K": k,
                        "retrieved_doc_ids": json.dumps([sd.doc_id for sd in scored_docs]),
                        "relevant_doc_ids": json.dumps(gold_doc_ids),
                        "ranked_scores": json.dumps([sd.score for sd in scored_docs]),
                        "latency_ms": latency_ms,
                        f"Recall@{k}": m.get(f"Recall@{k}", 0.0),
                        f"Precision@{k}": m.get(f"Precision@{k}", 0.0),
                        "MRR": m.get("MRR", 0.0),
                        f"Evidence Recall@{k}": m.get(f"EvidenceRecall@{k}", 0.0),
                        f"Evidence Completeness@{k}": m.get(f"EvidenceCompleteness@{k}", 0.0),
                    }
                )
                summary_accum[(method, k)].append(m)
                latency_accum[(method, k)].append(latency_ms)

    # -------------------------------------------------------------------------
    # Also load already-completed runs from file for summary computation
    # -------------------------------------------------------------------------
    if completed:
        logger.info(
            "Re-reading %d pre-existing run records for summary computation...",
            len(completed),
        )
        with open(raw_runs_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = (rec["method"], rec["k"])
                    # Only add if not already added in this session
                    tup = (rec["query_id"], rec["method"], rec["k"])
                    if tup in completed:
                        summary_accum[key].append(rec["metrics"])
                        latency_accum[key].append(rec["latency_ms"])
                        # Also build per_query_rows for CSV if missing
                        m = rec["metrics"]
                        k = rec["k"]
                        method = rec["method"]
                        per_query_rows.append(
                            {
                                "dataset": "hotpotqa",
                                "query_id": rec["query_id"],
                                "method": method,
                                "K": k,
                                "retrieved_doc_ids": json.dumps(rec.get("retrieved_doc_ids", [])),
                                "relevant_doc_ids": json.dumps(rec.get("relevant_doc_ids", [])),
                                "ranked_scores": json.dumps(rec.get("ranked_scores", [])),
                                "latency_ms": rec.get("latency_ms", 0.0),
                                f"Recall@{k}": m.get(f"Recall@{k}", 0.0),
                                f"Precision@{k}": m.get(f"Precision@{k}", 0.0),
                                "MRR": m.get("MRR", 0.0),
                                f"Evidence Recall@{k}": m.get(f"EvidenceRecall@{k}", 0.0),
                                f"Evidence Completeness@{k}": m.get(f"EvidenceCompleteness@{k}", 0.0),
                            }
                        )
                except (json.JSONDecodeError, KeyError):
                    continue

    # -------------------------------------------------------------------------
    # Write per-query CSV
    # -------------------------------------------------------------------------
    per_query_fieldnames = [
        "dataset", "query_id", "method", "K",
        "retrieved_doc_ids", "relevant_doc_ids", "ranked_scores", "latency_ms",
        "Recall@K", "Precision@K", "MRR", "Evidence Recall@K", "Evidence Completeness@K",
    ]

    with open(per_query_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_query_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in per_query_rows:
            k = row["K"]
            flat_row = {
                "dataset": row["dataset"],
                "query_id": row["query_id"],
                "method": row["method"],
                "K": k,
                "retrieved_doc_ids": row["retrieved_doc_ids"],
                "relevant_doc_ids": row["relevant_doc_ids"],
                "ranked_scores": row["ranked_scores"],
                "latency_ms": row["latency_ms"],
                f"Recall@K": row.get(f"Recall@{k}", 0.0),
                f"Precision@K": row.get(f"Precision@{k}", 0.0),
                "MRR": row["MRR"],
                f"Evidence Recall@K": row.get(f"Evidence Recall@{k}", 0.0),
                f"Evidence Completeness@K": row.get(f"Evidence Completeness@{k}", 0.0),
            }
            writer.writerow(flat_row)
    logger.info("Wrote per-query CSV to %s", per_query_csv_path)

    # -------------------------------------------------------------------------
    # Write summary CSV
    # -------------------------------------------------------------------------
    summary_rows: List[Dict[str, Any]] = []
    for (method, k), metric_list in sorted(summary_accum.items()):
        if not metric_list:
            continue
        n = len(metric_list)
        lat = latency_accum[(method, k)]

        def _mean(key: str) -> float:
            vals = [m.get(key, 0.0) for m in metric_list]
            return float(np.mean(vals))

        summary_rows.append(
            {
                "dataset": "hotpotqa",
                "method": method,
                "K": k,
                "query_count": n,
                f"Recall@K": _mean(f"Recall@{k}"),
                f"Precision@K": _mean(f"Precision@{k}"),
                "MRR": _mean("MRR"),
                f"Evidence Recall@K": _mean(f"EvidenceRecall@{k}"),
                f"Evidence Completeness@K": _mean(f"EvidenceCompleteness@{k}"),
                "mean_latency_ms": float(np.mean(lat)) if lat else 0.0,
                "median_latency_ms": float(np.median(lat)) if lat else 0.0,
            }
        )

    summary_fieldnames = [
        "dataset", "method", "K", "query_count",
        "Recall@K", "Precision@K", "MRR",
        "Evidence Recall@K", "Evidence Completeness@K",
        "mean_latency_ms", "median_latency_ms",
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info("Wrote summary CSV to %s", summary_csv_path)

    # Print summary table
    print("\n=== HotpotQA Retrieval Summary ===")
    header = f"{'Method':<10} {'K':>3}  {'N':>5}  {'R@K':>7}  {'P@K':>7}  {'MRR':>7}  {'EvR@K':>7}  {'EvC@K':>7}  {'mLat(ms)':>10}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['method']:<10} {row['K']:>3}  {row['query_count']:>5}  "
            f"{row['Recall@K']:>7.4f}  {row['Precision@K']:>7.4f}  "
            f"{row['MRR']:>7.4f}  {row['Evidence Recall@K']:>7.4f}  "
            f"{row['Evidence Completeness@K']:>7.4f}  {row['mean_latency_ms']:>10.2f}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HotpotQA distractor retrieval experiment."
    )
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bm25", "dense", "hybrid"],
        choices=["bm25", "dense", "hybrid"],
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[3, 5, 7, 9],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
    )
    parser.add_argument("--raw-runs", type=Path, default=RAW_RUNS_PATH)
    parser.add_argument("--per-query-csv", type=Path, default=PER_QUERY_CSV_PATH)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV_PATH)
    parser.add_argument("--model-id", type=str, default="BAAI/bge-base-en-v1.5")
    args = parser.parse_args()

    run_hotpotqa_experiment(
        methods=args.methods,
        k_values=args.k_values,
        processed_dir=args.processed_dir,
        raw_runs_path=args.raw_runs,
        per_query_csv_path=args.per_query_csv,
        summary_csv_path=args.summary_csv,
        model_id=args.model_id,
        overwrite=args.overwrite,
    )
