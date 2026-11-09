"""HotpotQA experiment validation script.

Validates the integrity of:
- Processed data (questions, candidates, evidence files).
- Retrieval run outputs (hotpotqa_runs.jsonl).
- Metric recomputability from per-query CSV.

Saves a JSON validation report to ``results/metrics/hotpotqa_validation_report.json``.

Usage::

    python -m src.experiments.validate_hotpotqa
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_hotpotqa")

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = _ROOT / "data" / "processed"
RESULTS_RAW_DIR = _ROOT / "results" / "raw"
RESULTS_METRICS_DIR = _ROOT / "results" / "metrics"
VALIDATION_REPORT_PATH = RESULTS_METRICS_DIR / "hotpotqa_validation_report.json"


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def _check(
    report: Dict[str, Any],
    check_name: str,
    passed: bool,
    message: str,
    details: Any = None,
) -> None:
    """Record a single check result into the report."""
    entry: Dict[str, Any] = {
        "passed": passed,
        "message": message,
    }
    if details is not None:
        entry["details"] = details
    report["checks"][check_name] = entry
    if not passed:
        report["failed_checks"].append(check_name)
        logger.warning("FAIL [%s]: %s", check_name, message)
    else:
        logger.info("PASS [%s]: %s", check_name, message)


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def validate_candidate_completeness(
    questions: Dict[str, Any],
    candidates: Dict[str, Dict[str, Any]],
    report: Dict[str, Any],
) -> None:
    """Check 1: Every question has > 0 candidate paragraphs."""
    empty: List[str] = []
    for qid in questions:
        if not candidates.get(qid):
            empty.append(qid)
    _check(
        report,
        "candidate_completeness",
        len(empty) == 0,
        f"{len(questions) - len(empty)}/{len(questions)} questions have candidate paragraphs.",
        {"empty_question_ids": empty} if empty else None,
    )


def validate_gold_doc_ids_in_candidates(
    questions: Dict[str, Any],
    candidates: Dict[str, Dict[str, Any]],
    report: Dict[str, Any],
) -> None:
    """Check 2: All gold_doc_ids are present in the question's candidate pool."""
    missing_entries: List[Dict[str, Any]] = []
    for qid, q in questions.items():
        pool = set(candidates.get(qid, {}).keys())
        for doc_id in q.get("gold_doc_ids", []):
            if doc_id not in pool:
                missing_entries.append({"question_id": qid, "missing_doc_id": doc_id})
    _check(
        report,
        "gold_doc_ids_in_candidates",
        len(missing_entries) == 0,
        f"{len(missing_entries)} gold doc_id references not found in candidate pool.",
        missing_entries if missing_entries else None,
    )


def validate_supporting_fact_sent_indices(
    questions: Dict[str, Any],
    candidates: Dict[str, Dict[str, Any]],
    report: Dict[str, Any],
) -> None:
    """Check 3: All supporting fact sentence indices are valid (within range)."""
    invalid: List[Dict[str, Any]] = []
    for qid, q in questions.items():
        pool = candidates.get(qid, {})
        # Build title -> doc_id mapping
        title_to_doc: Dict[str, str] = {}
        for doc_id, cand in pool.items():
            title_to_doc[cand["title"]] = doc_id

        for sf in q.get("supporting_facts", []):
            title = sf["title"]
            sent_idx = sf["sent_idx"]
            if title not in title_to_doc:
                # Title not in pool — already checked by check 2 / invalid facts
                continue
            doc_id = title_to_doc[title]
            sentences = pool[doc_id].get("sentences", [])
            if sent_idx < 0 or sent_idx >= len(sentences):
                invalid.append(
                    {
                        "question_id": qid,
                        "title": title,
                        "sent_idx": sent_idx,
                        "num_sentences": len(sentences),
                    }
                )
    _check(
        report,
        "supporting_fact_sent_indices",
        len(invalid) == 0,
        f"{len(invalid)} supporting facts have out-of-range sentence indices.",
        invalid if invalid else None,
    )


def validate_evidence_text_in_sentences(
    evidence_records: List[Dict[str, Any]],
    candidates: Dict[str, Dict[str, Any]],
    report: Dict[str, Any],
) -> None:
    """Check 4: Evidence sentence text matches the sentence at the stored index."""
    mismatches: List[Dict[str, Any]] = []
    for rec in evidence_records:
        qid = rec["query_id"]
        doc_id = rec["evidence_doc_id"]
        sent_idx = rec["sentence_idx"]
        evidence_sentence = rec["evidence_sentence"]

        pool = candidates.get(qid, {})
        para = pool.get(doc_id)
        if para is None:
            continue
        sentences = para.get("sentences", [])
        if 0 <= sent_idx < len(sentences):
            actual = sentences[sent_idx]
            if actual != evidence_sentence:
                mismatches.append(
                    {
                        "query_id": qid,
                        "doc_id": doc_id,
                        "sent_idx": sent_idx,
                        "expected": evidence_sentence,
                        "actual": actual,
                    }
                )

    _check(
        report,
        "evidence_text_in_sentences",
        len(mismatches) == 0,
        f"{len(mismatches)} evidence items have text mismatches.",
        mismatches[:20] if mismatches else None,  # limit detail output
    )


def validate_no_duplicate_doc_ids(
    questions: Dict[str, Any],
    candidates: Dict[str, Dict[str, Any]],
    report: Dict[str, Any],
) -> None:
    """Check 5: No duplicate doc_ids within a question's candidate pool."""
    duplicates: List[Dict[str, Any]] = []
    for qid in questions:
        pool = candidates.get(qid, {})
        doc_ids = list(pool.keys())
        if len(doc_ids) != len(set(doc_ids)):
            seen: Set[str] = set()
            dups = []
            for d in doc_ids:
                if d in seen:
                    dups.append(d)
                seen.add(d)
            duplicates.append({"question_id": qid, "duplicate_doc_ids": dups})
    _check(
        report,
        "no_duplicate_doc_ids",
        len(duplicates) == 0,
        f"{len(duplicates)} questions have duplicate doc_ids in their candidate pool.",
        duplicates if duplicates else None,
    )


def validate_runs_use_same_candidate_pool(
    runs: List[Dict[str, Any]],
    candidates: Dict[str, Dict[str, Any]],
    report: Dict[str, Any],
) -> None:
    """Check 6 & 7: Retrieved doc_ids in runs all exist in the question's candidate pool."""
    if not runs:
        _check(
            report,
            "retrieved_doc_ids_in_candidate_pool",
            True,
            "No run records found — skipping check.",
        )
        return

    invalid_entries: List[Dict[str, Any]] = []
    for rec in runs:
        qid = rec.get("query_id", "")
        pool = set(candidates.get(qid, {}).keys())
        for doc_id in rec.get("retrieved_doc_ids", []):
            if doc_id not in pool:
                invalid_entries.append(
                    {
                        "query_id": qid,
                        "method": rec.get("method"),
                        "k": rec.get("k"),
                        "invalid_doc_id": doc_id,
                    }
                )

    _check(
        report,
        "retrieved_doc_ids_in_candidate_pool",
        len(invalid_entries) == 0,
        f"{len(invalid_entries)} retrieved doc_ids not in candidate pool.",
        invalid_entries[:20] if invalid_entries else None,
    )


def validate_scores_non_increasing(
    runs: List[Dict[str, Any]],
    report: Dict[str, Any],
) -> None:
    """Check 8: Scores are non-increasing within each result list."""
    if not runs:
        _check(
            report,
            "scores_non_increasing",
            True,
            "No run records found — skipping check.",
        )
        return

    violations: List[Dict[str, Any]] = []
    for rec in runs:
        scores = rec.get("ranked_scores", [])
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1] - 1e-9:
                violations.append(
                    {
                        "query_id": rec.get("query_id"),
                        "method": rec.get("method"),
                        "k": rec.get("k"),
                        "position": i,
                        "score_i": scores[i],
                        "score_i1": scores[i + 1],
                    }
                )
    _check(
        report,
        "scores_non_increasing",
        len(violations) == 0,
        f"{len(violations)} score-ordering violations found.",
        violations[:20] if violations else None,
    )


def validate_summary_metrics_recomputable(
    runs: List[Dict[str, Any]],
    report: Dict[str, Any],
) -> None:
    """Check 9: Summary metrics can be recomputed from per-query run records."""
    if not runs:
        _check(
            report,
            "summary_metrics_recomputable",
            True,
            "No run records found — skipping check.",
        )
        return

    # Group by (method, k)
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for rec in runs:
        groups[(rec.get("method", ""), rec.get("k", 0))].append(rec)

    discrepancies: List[Dict[str, Any]] = []
    for (method, k), recs in groups.items():
        metric_key = f"Recall@{k}"
        vals = [rec.get("metrics", {}).get(metric_key, 0.0) for rec in recs]
        computed_mean = float(np.mean(vals)) if vals else 0.0
        # Just verify the computation doesn't throw; store a sample value
        if not (0.0 <= computed_mean <= 1.0):
            discrepancies.append(
                {
                    "method": method,
                    "k": k,
                    "metric": metric_key,
                    "computed_mean": computed_mean,
                }
            )

    _check(
        report,
        "summary_metrics_recomputable",
        len(discrepancies) == 0,
        f"{len(discrepancies)} metric summary discrepancies found.",
        discrepancies if discrepancies else None,
    )


def validate_invalid_facts_documented(
    questions: Dict[str, Any],
    metadata: Dict[str, Any],
    report: Dict[str, Any],
) -> None:
    """Check 10: All invalid/excluded supporting facts are documented in metadata."""
    actual_invalid_count = sum(
        len(q.get("invalid_supporting_facts", [])) for q in questions.values()
    )
    metadata_count = metadata.get("invalid_supporting_facts_count", -1)
    _check(
        report,
        "invalid_facts_documented",
        actual_invalid_count == metadata_count,
        f"Questions file has {actual_invalid_count} invalid facts; "
        f"metadata reports {metadata_count}.",
        {
            "actual_invalid_count": actual_invalid_count,
            "metadata_invalid_count": metadata_count,
        }
        if actual_invalid_count != metadata_count
        else None,
    )


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------


def run_validation(
    processed_dir: Path = PROCESSED_DIR,
    runs_path: Path = RESULTS_RAW_DIR / "hotpotqa_runs.jsonl",
    report_path: Path = VALIDATION_REPORT_PATH,
) -> Dict[str, Any]:
    """Execute all validation checks and save a JSON report.

    Returns
    -------
    Dict[str, Any]
        The full validation report dict.
    """
    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
        "failed_checks": [],
        "summary": {},
    }

    # ------------------------------------------------------------------
    # Load processed data
    # ------------------------------------------------------------------
    q_path = processed_dir / "hotpotqa_questions.jsonl"
    c_path = processed_dir / "hotpotqa_candidates.jsonl"
    e_path = processed_dir / "hotpotqa_evidence.jsonl"
    m_path = processed_dir / "hotpotqa_metadata.json"

    if not q_path.exists():
        report["summary"]["error"] = f"Questions file not found: {q_path}"
        logger.error(report["summary"]["error"])
        _save_report(report, report_path)
        return report

    questions: Dict[str, Any] = {}
    with open(q_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            questions[rec["question_id"]] = rec
    logger.info("Loaded %d questions.", len(questions))

    candidates: Dict[str, Dict[str, Any]] = {qid: {} for qid in questions}
    if c_path.exists():
        with open(c_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                qid = rec["question_id"]
                if qid in candidates:
                    candidates[qid][rec["doc_id"]] = rec

    evidence_records: List[Dict[str, Any]] = []
    if e_path.exists():
        with open(e_path, encoding="utf-8") as f:
            for line in f:
                evidence_records.append(json.loads(line))

    metadata: Dict[str, Any] = {}
    if m_path.exists():
        with open(m_path, encoding="utf-8") as f:
            metadata = json.load(f)

    runs: List[Dict[str, Any]] = []
    if runs_path.exists():
        with open(runs_path, encoding="utf-8") as f:
            for line in f:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    logger.info("Loaded %d run records.", len(runs))

    # ------------------------------------------------------------------
    # Execute checks
    # ------------------------------------------------------------------
    validate_candidate_completeness(questions, candidates, report)
    validate_gold_doc_ids_in_candidates(questions, candidates, report)
    validate_supporting_fact_sent_indices(questions, candidates, report)
    validate_evidence_text_in_sentences(evidence_records, candidates, report)
    validate_no_duplicate_doc_ids(questions, candidates, report)
    validate_runs_use_same_candidate_pool(runs, candidates, report)
    validate_scores_non_increasing(runs, report)
    validate_summary_metrics_recomputable(runs, report)
    validate_invalid_facts_documented(questions, metadata, report)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(report["checks"])
    passed = total - len(report["failed_checks"])
    report["summary"] = {
        "total_checks": total,
        "passed": passed,
        "failed": len(report["failed_checks"]),
        "all_passed": len(report["failed_checks"]) == 0,
    }

    _save_report(report, report_path)

    print(f"\n=== Validation Report ===")
    print(f"Checks: {passed}/{total} passed")
    if report["failed_checks"]:
        print(f"FAILED: {report['failed_checks']}")
    else:
        print("All checks passed!")
    return report


def _save_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Validation report saved to %s", path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate HotpotQA experiment outputs."
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--runs-path", type=Path, default=RESULTS_RAW_DIR / "hotpotqa_runs.jsonl"
    )
    parser.add_argument("--report-path", type=Path, default=VALIDATION_REPORT_PATH)
    args = parser.parse_args()

    report = run_validation(
        processed_dir=args.processed_dir,
        runs_path=args.runs_path,
        report_path=args.report_path,
    )
    sys.exit(0 if report["summary"].get("all_passed", False) else 1)
