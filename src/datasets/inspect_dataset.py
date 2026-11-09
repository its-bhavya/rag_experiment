"""Dataset inspection script reporting structural integrity and statistics.

Reports:
- Corpus count
- Query count
- Qrels count
- Empty documents
- Duplicate IDs
- Missing query IDs / doc IDs in qrels
- Document length distribution (word count and character count)
- Evidence annotation distribution
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from src.datasets.scifact import (
    load_scifact_dataset,
    process_and_save_scifact,
)


def inspect_scifact(
    data_dir: Path | str = "data/raw/scifact",
    split: str = "test",
) -> Dict[str, Any]:
    """Inspect SciFact dataset and return a detailed diagnostic dictionary."""
    dataset = load_scifact_dataset(data_dir=data_dir, split=split)

    corpus = dataset.corpus
    queries = dataset.queries
    qrels = dataset.qrels

    # 1. Counts
    corpus_count = len(corpus)
    query_count = len(queries)
    qrels_query_count = len(qrels)
    qrels_pair_count = sum(len(docs) for docs in qrels.values())

    # 2. Empty documents
    empty_docs: List[str] = []
    char_lengths: List[int] = []
    word_lengths: List[int] = []

    for doc_id, doc in corpus.items():
        text = doc.text.strip()
        title = doc.title.strip()
        full = doc.full_text
        if not text and not title:
            empty_docs.append(doc_id)
        char_lengths.append(len(full))
        word_lengths.append(len(full.split()))

    # 3. Duplicate checks (parse_corpus and parse_queries strictly raise on duplicates)
    # We also verify no duplicate keys exist in the dictionaries
    duplicate_doc_ids = corpus_count != len(set(corpus.keys()))
    duplicate_query_ids = query_count != len(set(queries.keys()))

    # 4. Missing IDs in Qrels
    missing_queries_in_qrels: List[str] = []
    missing_docs_in_qrels: List[str] = []
    corpus_id_set = set(corpus.keys())
    query_id_set = set(queries.keys())

    for qid, doc_dict in qrels.items():
        if qid not in query_id_set:
            missing_queries_in_qrels.append(qid)
        for doc_id in doc_dict.keys():
            if doc_id not in corpus_id_set:
                missing_docs_in_qrels.append(doc_id)

    # 5. Document length distribution stats
    def compute_stats(arr: List[int]) -> Dict[str, float]:
        if not arr:
            return {}
        a = np.array(arr)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)),
            "max": float(np.max(a)),
        }

    word_stats = compute_stats(word_lengths)
    char_stats = compute_stats(char_lengths)

    # 6. Evidence annotations stats
    queries_with_evidence = sum(1 for q in queries.values() if q.evidence)
    total_evidence_units = sum(len(q.evidence) for q in queries.values())
    test_queries_with_evidence = sum(
        1 for qid in qrels if qid in queries and queries[qid].evidence
    )
    test_evidence_units = sum(
        len(queries[qid].evidence) for qid in qrels if qid in queries
    )

    report = {
        "dataset": "SciFact",
        "split": split,
        "counts": {
            "corpus_count": corpus_count,
            "query_count": query_count,
            "qrels_queries_count": qrels_query_count,
            "qrels_pairs_count": qrels_pair_count,
        },
        "integrity": {
            "empty_documents_count": len(empty_docs),
            "empty_documents": empty_docs,
            "duplicate_doc_ids_found": duplicate_doc_ids,
            "duplicate_query_ids_found": duplicate_query_ids,
            "missing_queries_in_qrels_count": len(missing_queries_in_qrels),
            "missing_queries_in_qrels": missing_queries_in_qrels,
            "missing_docs_in_qrels_count": len(missing_docs_in_qrels),
            "missing_docs_in_qrels": missing_docs_in_qrels,
        },
        "length_distribution": {
            "word_count": word_stats,
            "char_count": char_stats,
        },
        "evidence_annotations": {
            "all_queries_with_evidence": queries_with_evidence,
            "all_queries_total_evidence_units": total_evidence_units,
            "test_queries_with_evidence": test_queries_with_evidence,
            "test_queries_total_evidence_units": test_evidence_units,
        },
    }

    return report


def format_report(report: Dict[str, Any]) -> str:
    """Format the diagnostic inspection dictionary into a clean CLI string."""
    c = report["counts"]
    integ = report["integrity"]
    w = report["length_distribution"]["word_count"]
    ev = report["evidence_annotations"]

    lines = [
        "=" * 65,
        f"DATASET INSPECTION REPORT: {report['dataset']} (Split: {report['split']})",
        "=" * 65,
        "1. RECORD COUNTS",
        f"  - Corpus Documents:       {c['corpus_count']:,}",
        f"  - Total Queries:          {c['query_count']:,}",
        f"  - Qrels Queries ({report['split']}):   {c['qrels_queries_count']:,}",
        f"  - Qrels Relevance Pairs:  {c['qrels_pairs_count']:,}",
        "",
        "2. INTEGRITY CHECKS",
        f"  - Empty Documents:        {integ['empty_documents_count']} (IDs: {integ['empty_documents']})",
        f"  - Duplicate Doc IDs:      {integ['duplicate_doc_ids_found']}",
        f"  - Duplicate Query IDs:    {integ['duplicate_query_ids_found']}",
        f"  - Missing Query IDs in Qrels: {integ['missing_queries_in_qrels_count']}",
        f"  - Missing Doc IDs in Qrels:   {integ['missing_docs_in_qrels_count']}",
        "",
        "3. DOCUMENT LENGTH DISTRIBUTION (Word Counts)",
        f"  - Min / Max:              {w['min']:.0f} / {w['max']:.0f} words",
        f"  - Mean +/- Std:           {w['mean']:.1f} +/- {w['std']:.1f} words",
        f"  - 25th / Median / 75th:   {w['p25']:.1f} / {w['median']:.1f} / {w['p75']:.1f} words",
        f"  - 95th Percentile:        {w['p95']:.1f} words",
        "",
        "4. SUPPORTING EVIDENCE ANNOTATIONS",
        f"  - Total Claims with Evidence:      {ev['all_queries_with_evidence']:,} / {c['query_count']:,}",
        f"  - Total Evidence Units (Sentences):{ev['all_queries_total_evidence_units']:,}",
        f"  - Test Claims with Evidence:       {ev['test_queries_with_evidence']:,} / {c['qrels_queries_count']:,}",
        f"  - Test Evidence Units:             {ev['test_queries_total_evidence_units']:,}",
        "=" * 65,
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset integrity and statistics.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/scifact",
        help="Path to raw dataset directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (e.g. test, train)",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Also export processed dataset to data/processed/scifact",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/raw/scifact_inspection.json",
        help="Path to save inspection JSON report",
    )

    args = parser.parse_args()

    report = inspect_scifact(data_dir=args.data_dir, split=args.split)
    print(format_report(report))

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved machine-readable inspection to {out_p}")

    if args.process:
        print("\nProcessing and saving clean artifacts to data/processed/scifact...")
        process_and_save_scifact(raw_dir=args.data_dir, split=args.split)
        print("Done.")


if __name__ == "__main__":
    main()
