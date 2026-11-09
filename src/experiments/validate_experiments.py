"""Comprehensive validation and quality checks for retrieval experiments.

Verifies 10 critical integrity requirements for scientific reproducibility:
1. Every query has the expected qrels entry or is explicitly marked as having no relevant documents.
2. Every retrieved document ID exists in the corpus.
3. Every evidence annotation refers to a valid query.
4. Evidence annotations are mapped correctly to corpus text.
5. Retrieval results are sorted by score/rank monotonically.
6. No duplicate document IDs occur in a result list.
7. Metrics are within valid mathematical ranges ([0.0, 1.0]).
8. Every experiment configuration has the exact same query count.
9. Missing evidence annotations are not silently counted as evidence retrieval failures.
10. All summary metrics can be recomputed from the per-query results.

Output:
    results/metrics/validation_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(_ROOT) not in sys.path:
    import sys
    sys.path.insert(0, str(_ROOT))

from src.datasets.scifact import parse_corpus, parse_queries, parse_qrels
from src.evaluation.metrics import matches_evidence_text

logger = logging.getLogger("validate_experiments")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

CORPUS_PATH = _ROOT / "data" / "raw" / "scifact" / "corpus.jsonl"
QUERIES_PATH = _ROOT / "data" / "raw" / "scifact" / "queries.jsonl"
QRELS_PATH = _ROOT / "data" / "raw" / "scifact" / "qrels" / "test.tsv"
EVIDENCE_JSONL_PATH = _ROOT / "data" / "processed" / "scifact_evidence.jsonl"
EVIDENCE_REPORT_PATH = _ROOT / "data" / "processed" / "evidence_mapping_report.json"

RUNS_PATH = _ROOT / "results" / "raw" / "scifact_runs.jsonl"
PER_QUERY_CSV_PATH = _ROOT / "results" / "metrics" / "scifact_per_query.csv"
SUMMARY_CSV_PATH = _ROOT / "results" / "metrics" / "scifact_summary.csv"
VALIDATION_REPORT_PATH = _ROOT / "results" / "metrics" / "validation_report.json"


def check_1_qrels_coverage(
    eval_records: List[Dict[str, Any]],
    qrels: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    """1. Every query has expected qrels entry or is explicitly marked as having no relevant docs."""
    eval_qids = {str(r["query_id"]) for r in eval_records}
    missing_in_qrels: List[str] = []
    unmarked_empty: List[str] = []

    for r in eval_records:
        qid = str(r["query_id"])
        rel_ids = r.get("relevant_document_ids")
        if qid not in qrels:
            # Must be explicitly empty list
            if rel_ids is None or len(rel_ids) > 0:
                missing_in_qrels.append(qid)
        else:
            gold_qrel_set = set(qrels[qid].keys())
            rec_rel_set = set(rel_ids) if rel_ids is not None else set()
            if rec_rel_set != gold_qrel_set:
                missing_in_qrels.append(qid)

    passed = len(missing_in_qrels) == 0 and len(unmarked_empty) == 0
    return {
        "passed": passed,
        "total_queries_checked": len(eval_qids),
        "discrepant_qrels_queries": list(set(missing_in_qrels)),
        "details": f"All {len(eval_qids)} evaluated queries strictly align with official qrels."
        if passed
        else f"{len(missing_in_qrels)} queries had qrels discrepancies.",
    }


def check_2_corpus_id_validity(
    eval_records: List[Dict[str, Any]],
    corpus_ids: Set[str],
) -> Dict[str, Any]:
    """2. Every retrieved document ID exists in the corpus."""
    invalid_ids: Set[str] = set()
    total_retrieved_checked = 0

    for r in eval_records:
        for doc_id in r.get("retrieved_document_ids", []):
            total_retrieved_checked += 1
            if str(doc_id) not in corpus_ids:
                invalid_ids.add(str(doc_id))

    passed = len(invalid_ids) == 0
    return {
        "passed": passed,
        "total_retrieved_docs_checked": total_retrieved_checked,
        "invalid_document_ids_count": len(invalid_ids),
        "invalid_document_ids": list(invalid_ids)[:10],
        "details": f"All {total_retrieved_checked} retrieved document IDs exist in the corpus."
        if passed
        else f"{len(invalid_ids)} unrecognized document IDs found.",
    }


def check_3_evidence_query_validity(
    evidence_records: List[Dict[str, Any]],
    all_query_ids: Set[str],
) -> Dict[str, Any]:
    """3. Every evidence annotation refers to a valid query."""
    invalid_qids: Set[str] = set()
    for ev in evidence_records:
        qid = str(ev.get("query_id", ""))
        if not qid or qid not in all_query_ids:
            invalid_qids.add(qid)

    passed = len(invalid_qids) == 0
    return {
        "passed": passed,
        "total_evidence_records_checked": len(evidence_records),
        "invalid_query_ids_count": len(invalid_qids),
        "invalid_query_ids": list(invalid_qids),
        "details": f"All {len(evidence_records)} evidence records map to valid queries."
        if passed
        else f"{len(invalid_qids)} evidence records have invalid query IDs.",
    }


def check_4_evidence_text_mapping(
    evidence_records: List[Dict[str, Any]],
    corpus: Dict[str, Any],
) -> Dict[str, Any]:
    """4. Evidence annotations are mapped correctly to corpus text."""
    failed_matches: List[Dict[str, Any]] = []
    matched_count = 0

    for ev in evidence_records:
        if ev.get("evidence_match_status") != "matched":
            continue
        matched_count += 1
        doc_id = str(ev.get("evidence_doc_id", ""))
        sent_text = ev.get("evidence_sentence", "")

        if doc_id not in corpus:
            failed_matches.append({"evidence_id": ev.get("evidence_annotation_id"), "reason": "doc_id_missing"})
            continue

        corpus_text = corpus[doc_id].full_text
        if not matches_evidence_text(sent_text, corpus_text):
            failed_matches.append(
                {
                    "evidence_id": ev.get("evidence_annotation_id"),
                    "doc_id": doc_id,
                    "reason": "text_not_found_in_corpus_doc",
                }
            )

    passed = len(failed_matches) == 0
    return {
        "passed": passed,
        "total_matched_evidence_checked": matched_count,
        "failed_matches_count": len(failed_matches),
        "failed_matches": failed_matches[:5],
        "details": f"All {matched_count} mapped evidence sentences exist verbatim or normalized in their corpus document."
        if passed
        else f"{len(failed_matches)} evidence sentences could not be verified in corpus text.",
    }


def check_5_score_monotonicity(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """5. Retrieval results are sorted by score/rank monotonically."""
    non_monotonic: List[str] = []
    for r in eval_records:
        scores = r.get("ranked_scores", [])
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1]:
                non_monotonic.append(f"{r['method']}_q{r['query_id']}_K{r['K']}")
                break

    passed = len(non_monotonic) == 0
    return {
        "passed": passed,
        "total_runs_checked": len(eval_records),
        "non_monotonic_runs_count": len(non_monotonic),
        "details": "All retrieved result lists are strictly sorted by score non-increasingly."
        if passed
        else f"{len(non_monotonic)} runs contained non-monotonic score orders.",
    }


def check_6_no_duplicate_retrieved_ids(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """6. No duplicate document IDs occur in a result list."""
    duplicate_runs: List[str] = []
    for r in eval_records:
        doc_ids = r.get("retrieved_document_ids", [])
        if len(doc_ids) != len(set(doc_ids)):
            duplicate_runs.append(f"{r['method']}_q{r['query_id']}_K{r['K']}")

    passed = len(duplicate_runs) == 0
    return {
        "passed": passed,
        "total_runs_checked": len(eval_records),
        "duplicate_runs_count": len(duplicate_runs),
        "details": "All result lists contain strictly unique document IDs (no duplicates)."
        if passed
        else f"{len(duplicate_runs)} result lists contained duplicate document IDs.",
    }


def check_7_metric_ranges(df_per_query: pd.DataFrame) -> Dict[str, Any]:
    """7. Metrics are within valid mathematical ranges ([0.0, 1.0])."""
    violations: List[str] = []

    for metric in ["Recall@K", "Precision@K", "MRR"]:
        vals = df_per_query[metric].dropna()
        if (vals < 0.0).any() or (vals > 1.0).any():
            violations.append(f"{metric} out of [0, 1]")

    for metric in ["EvidenceRecall@K", "EvidenceCompleteness@K"]:
        vals = df_per_query[metric].dropna()
        if (vals < 0.0).any() or (vals > 1.0).any():
            violations.append(f"{metric} out of [0, 1]")
        if metric == "EvidenceCompleteness@K":
            # Must be strictly 0.0 or 1.0
            invalid_binary = vals[~vals.isin([0.0, 1.0])]
            if not invalid_binary.empty:
                violations.append("EvidenceCompleteness@K contains non-binary values")

    # Latency check
    lat = df_per_query["latency_ms"].dropna()
    if (lat < 0.0).any():
        violations.append("Negative latency found")

    passed = len(violations) == 0
    return {
        "passed": passed,
        "violations": violations,
        "details": "All metric values strictly adhere to valid ranges: Recall, Precision, MRR, EvidenceRecall in [0, 1], EvidenceCompleteness in {0, 1}, and Latency >= 0."
        if passed
        else f"Violations found: {violations}",
    }


def check_8_consistent_query_counts(df_summary: pd.DataFrame) -> Dict[str, Any]:
    """8. Every experiment configuration has the exact same query count."""
    counts = df_summary["query_count"].unique()
    passed = len(counts) == 1
    return {
        "passed": bool(passed),
        "query_counts_found": [int(c) for c in counts],
        "details": f"All {len(df_summary)} experimental configurations evaluated the exact same query count ({counts[0]})."
        if passed
        else f"Inconsistent query counts across configurations: {counts}",
    }


def check_9_missing_evidence_not_penalized(df_per_query: pd.DataFrame) -> Dict[str, Any]:
    """9. Missing evidence annotations are not silently counted as evidence retrieval failures."""
    # Queries without evidence annotations must have NaN/empty for evidence metrics
    no_ev_subset = df_per_query[df_per_query["num_evidence_annotated"].isna()]
    ev_recall_vals = no_ev_subset["EvidenceRecall@K"].dropna()
    ev_comp_vals = no_ev_subset["EvidenceCompleteness@K"].dropna()

    passed = ev_recall_vals.empty and ev_comp_vals.empty
    return {
        "passed": bool(passed),
        "unannotated_queries_count": len(no_ev_subset),
        "unannotated_queries_with_falsified_zeros": len(ev_recall_vals) + len(ev_comp_vals),
        "details": "Queries without evidence annotations correctly have null/NaN evidence metrics and are not penalized with artificial zeros."
        if passed
        else "Queries without evidence annotations were penalized with zeros!",
    }


def check_10_summary_recomputability(
    df_per_query: pd.DataFrame,
    df_summary: pd.DataFrame,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """10. All summary metrics can be recomputed from the per-query results."""
    mismatches: List[Dict[str, Any]] = []

    for _, s_row in df_summary.iterrows():
        method = s_row["method"]
        k = s_row["K"]
        subset = df_per_query[(df_per_query["method"] == method) & (df_per_query["K"] == k)]

        recomputed = {
            "query_count": len(subset),
            "Recall@K": round(float(subset["Recall@K"].mean()), 4),
            "Precision@K": round(float(subset["Precision@K"].mean()), 4),
            "MRR": round(float(subset["MRR"].mean()), 4),
            "Evidence Recall@K": round(float(subset["EvidenceRecall@K"].dropna().mean()), 4),
            "Evidence Completeness@K": round(float(subset["EvidenceCompleteness@K"].dropna().mean()), 4),
            "mean latency": round(float(subset["latency_ms"].mean()), 3),
            "median latency": round(float(subset["latency_ms"].median()), 3),
        }

        for metric_name, recomp_val in recomputed.items():
            reported_val = s_row[metric_name]
            if not np.isclose(recomp_val, reported_val, atol=atol):
                mismatches.append(
                    {
                        "method": method,
                        "K": k,
                        "metric": metric_name,
                        "reported": reported_val,
                        "recomputed": recomp_val,
                    }
                )

    passed = len(mismatches) == 0
    return {
        "passed": passed,
        "mismatches_count": len(mismatches),
        "mismatches": mismatches,
        "details": "100% of summary metrics across all configurations were independently recomputed and match per-query averages."
        if passed
        else f"{len(mismatches)} summary metric mismatches detected.",
    }


def run_all_validation_checks() -> Dict[str, Any]:
    """Execute all 10 validation checks and return the complete validation report dictionary."""
    logger.info("=== Starting Validation Suite (10 Comprehensive Quality Checks) ===")

    # Load resources
    logger.info("Loading corpus, queries, qrels, and runs...")
    corpus = parse_corpus(CORPUS_PATH)
    corpus_ids = set(corpus.keys())
    queries = parse_queries(QUERIES_PATH)
    all_query_ids = set(queries.keys())
    qrels = parse_qrels(QRELS_PATH)

    evidence_records: List[Dict[str, Any]] = []
    with open(EVIDENCE_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                evidence_records.append(json.loads(line))

    eval_records: List[Dict[str, Any]] = []
    with open(RUNS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_records.append(json.loads(line))

    df_per_query = pd.read_csv(PER_QUERY_CSV_PATH)
    df_summary = pd.read_csv(SUMMARY_CSV_PATH)

    # Execute all 10 checks
    logger.info("Check 1: Qrels coverage...")
    c1 = check_1_qrels_coverage(eval_records, qrels)

    logger.info("Check 2: Corpus ID validity...")
    c2 = check_2_corpus_id_validity(eval_records, corpus_ids)

    logger.info("Check 3: Evidence query validity...")
    c3 = check_3_evidence_query_validity(evidence_records, all_query_ids)

    logger.info("Check 4: Evidence text mapping...")
    c4 = check_4_evidence_text_mapping(evidence_records, corpus)

    logger.info("Check 5: Score monotonicity...")
    c5 = check_5_score_monotonicity(eval_records)

    logger.info("Check 6: No duplicate retrieved document IDs...")
    c6 = check_6_no_duplicate_retrieved_ids(eval_records)

    logger.info("Check 7: Metric ranges...")
    c7 = check_7_metric_ranges(df_per_query)

    logger.info("Check 8: Consistent query counts...")
    c8 = check_8_consistent_query_counts(df_summary)

    logger.info("Check 9: Missing evidence annotations not penalized...")
    c9 = check_9_missing_evidence_not_penalized(df_per_query)

    logger.info("Check 10: Summary recomputability...")
    c10 = check_10_summary_recomputability(df_per_query, df_summary)

    checks = {
        "1_qrels_coverage": c1,
        "2_corpus_id_validity": c2,
        "3_evidence_query_validity": c3,
        "4_evidence_text_mapping": c4,
        "5_score_monotonicity": c5,
        "6_no_duplicate_retrieved_ids": c6,
        "7_metric_ranges": c7,
        "8_consistent_query_counts": c8,
        "9_missing_evidence_not_penalized": c9,
        "10_summary_recomputability": c10,
    }

    all_passed = all(c["passed"] for c in checks.values())
    passed_count = sum(1 for c in checks.values() if c["passed"])

    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "scifact",
        "validation_passed": all_passed,
        "summary": {
            "total_checks": len(checks),
            "passed_checks": passed_count,
            "failed_checks": len(checks) - passed_count,
        },
        "safe_to_use_in_paper": bool(all_passed),
        "checks": checks,
    }

    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved validation report to {VALIDATION_REPORT_PATH}")
    logger.info(f"Validation status: {'PASSED (All 10/10 checks passed)' if all_passed else 'FAILED'}")
    logger.info(f"Safe to use in paper: {report['safe_to_use_in_paper']}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment validation and quality checks.")
    parser.parse_args()
    report = run_all_validation_checks()
    print("\n" + "=" * 80)
    print("                      VALIDATION REPORT SUMMARY")
    print("=" * 80)
    for check_name, res in report["checks"].items():
        status = "PASSED" if res["passed"] else "FAILED"
        print(f"[{status}] {check_name:<36} : {res['details']}")
    print("-" * 80)
    print(f"OVERALL STATUS     : {'PASSED' if report['validation_passed'] else 'FAILED'}")
    print(f"SAFE FOR PAPER     : {report['safe_to_use_in_paper']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
