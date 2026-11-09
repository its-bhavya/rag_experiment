"""Audit computation script for SciFact experiment.
Generates a detailed report comparing independently recomputed metrics
with the existing summary CSV and answers audit questions.
"""
import json
import csv
from pathlib import Path
from collections import defaultdict

# Project root
_ROOT = Path(__file__).resolve().parent.parent.parent

# Import metric utilities
from src.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    # MRR function is defined later, import via module
    )

# MRR implementation is inside metrics.py as mrr_at_k? Let's inspect file to find name.
# We'll import the whole module and locate function.
from src.evaluation import metrics as m

def read_jsonl(path):
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_runs():
    runs_path = _ROOT / "results" / "raw" / "scifact_runs.jsonl"
    runs = list(read_jsonl(runs_path))
    return runs

def load_summary_csv():
    summary_path = _ROOT / "results" / "metrics" / "scifact_summary.csv"
    rows = []
    with open(summary_path, newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def load_per_query_csv():
    per_path = _ROOT / "results" / "metrics" / "scifact_per_query.csv"
    rows = []
    with open(per_path, newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def compute_metrics(runs):
    # Group runs by (method, K, query_id)
    grouped = defaultdict(list)
    for rec in runs:
        key = (rec["method"], rec["K"], rec["query_id"])
        grouped[key].append(rec)
    # For each group compute metrics using the stored fields
    results = []
    for (method, K, qid), records in grouped.items():
        # Assuming each record corresponds to a single run with fields:
        # retrieved_document_ids, relevant_document_ids, evidence_units_retrieved, etc.
        # Take first (they should be identical per run)
        rec = records[0]
        retrieved = rec.get("retrieved_document_ids", [])
        relevant = rec.get("relevant_document_ids", [])
        # Document metrics
        recall = recall_at_k(retrieved, relevant, K, allow_duplicates=False, empty_relevant_handling="zero")
        precision = precision_at_k(retrieved, relevant, K, allow_duplicates=False)
        # MRR
        mrr = m.mrr_at_k(retrieved, relevant, K)
        # Evidence metrics – use stored counts if present else compute placeholder 0
        ev_retrieved = rec.get("evidence_units_retrieved", 0)
        ev_total = rec.get("evidence_units_total", 0)
        ev_recall = ev_retrieved / ev_total if ev_total else 0.0
        ev_completeness = 1.0 if ev_recall == 1.0 and ev_total>0 else 0.0
        results.append({
            "method": method,
            "K": K,
            "query_id": qid,
            "Recall@K": recall,
            "Precision@K": precision,
            "MRR": mrr,
            "EvidenceRecall@K": ev_recall,
            "EvidenceCompleteness@K": ev_completeness,
        })
    return results

def aggregate_summary(per_query_results):
    # Aggregate macro averages per (method, K)
    agg = defaultdict(list)
    for r in per_query_results:
        agg[(r["method"], r["K"])].append(r)
    summary = []
    for (method, K), lst in agg.items():
        n = len(lst)
        summary.append({
            "method": method,
            "K": K,
            "query_count": n,
            "Recall@K": sum(x["Recall@K"] for x in lst) / n,
            "Precision@K": sum(x["Precision@K"] for x in lst) / n,
            "MRR": sum(x["MRR"] for x in lst) / n,
            "EvidenceRecall@K": sum(x["EvidenceRecall@K"] for x in lst) / n,
            "EvidenceCompleteness@K": sum(x["EvidenceCompleteness@K"] for x in lst) / n,
        })
    return summary

def main():
    runs = load_runs()
    per_query = load_per_query_csv()
    summary = load_summary_csv()
    # Compute independent metrics
    computed_per = compute_metrics(runs)
    computed_summary = aggregate_summary(computed_per)
    # Helper to compare two lists of dicts (order may differ)
    def index_rows(rows, keys):
        return {tuple(row[k] for k in keys): row for row in rows}
    per_index = index_rows(per_query, ["method", "K", "query_id"])
    comp_per_index = index_rows(computed_per, ["method", "K", "query_id"])
    per_mismatches = []
    for key, row in per_index.items():
        comp = comp_per_index.get(key)
        if not comp:
            per_mismatches.append({"key": key, "issue": "missing in computed"})
            continue
        # compare a few metric columns within tolerance
        for metric in ["Recall@K", "Precision@K", "MRR", "EvidenceRecall@K", "EvidenceCompleteness@K"]:
            a = float(row[metric])
            b = float(comp[metric])
            if abs(a - b) > 1e-4:
                per_mismatches.append({"key": key, "metric": metric, "csv": a, "computed": b})
    # summary comparison
    sum_index = index_rows(summary, ["method", "K"])
    comp_sum_index = index_rows(computed_summary, ["method", "K"])
    summary_mismatches = []
    for key, row in sum_index.items():
        comp = comp_sum_index.get(key)
        if not comp:
            summary_mismatches.append({"key": key, "issue": "missing"})
            continue
        for metric in ["Recall@K", "Precision@K", "MRR", "EvidenceRecall@K", "EvidenceCompleteness@K"]:
            a = float(row[metric])
            b = float(comp[metric])
            if abs(a - b) > 1e-4:
                summary_mismatches.append({"key": key, "metric": metric, "csv": a, "computed": b})
    # Answer audit questions using available data
    # 1. Number of SciFact test queries evaluated
    eval_queries = {str(r["query_id"]) for r in runs}
    num_test_queries = len(eval_queries)
    # 2. Queries included in document metrics (those with qrels)
    doc_metric_queries = {str(r["query_id"]) for r in runs if r.get("relevant_document_ids")}
    num_doc_metric_queries = len(doc_metric_queries)
    # 3. Queries included in evidence metrics (those with evidence annotations)
    ev_metric_queries = {str(r["query_id"]) for r in runs if r.get("evidence_units_total")}
    num_ev_metric_queries = len(ev_metric_queries)
    # 4. Supporting sentences/evidence units annotated
    evidence_path = _ROOT / "data" / "processed" / "scifact_evidence.jsonl"
    total_evidence = sum(1 for _ in read_jsonl(evidence_path))
    # 5. Supporting evidence units successfully mapped (matched)
    matched_evidence = sum(1 for ev in read_jsonl(evidence_path) if ev.get("evidence_match_status") == "matched")
    # 6-14 are textual answers – we will embed them later.
    report = {
        "num_test_queries": num_test_queries,
        "num_doc_metric_queries": num_doc_metric_queries,
        "num_ev_metric_queries": num_ev_metric_queries,
        "total_evidence_units": total_evidence,
        "mapped_evidence_units": matched_evidence,
        "per_query_mismatches": per_mismatches,
        "summary_mismatches": summary_mismatches,
        "validation_report": json.load(open(_ROOT / "results" / "metrics" / "validation_report.json")),
    }
    # Write JSON report for debugging
    out_path = _ROOT / "results" / "metrics" / "audit_report.json"
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2)
    print("AUDIT DONE", out_path)

if __name__ == "__main__":
    main()
