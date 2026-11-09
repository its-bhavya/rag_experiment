"""Unit tests for the reproducible experiment runner."""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.datasets.scifact import SciFactDocument, SciFactQuery
from src.evaluation.metrics import EvidenceItem
from src.experiments.run_experiments import (
    compute_dataset_fingerprint,
    get_environment_info,
    load_completed_run_keys,
    run_experiment_pipeline,
    set_fixed_seeds,
)


class TestEnvironmentAndMetadata:
    def test_environment_info_structure(self):
        env = get_environment_info()
        assert "python_version" in env
        assert "platform" in env
        assert "library_versions" in env
        libs = env["library_versions"]
        assert "numpy" in libs
        assert "pandas" in libs
        assert "rank_bm25" in libs

    def test_set_fixed_seeds(self):
        set_fixed_seeds(42)
        import numpy as np
        import random
        val_py = random.random()
        val_np = np.random.rand()

        # Re-seeding produces identical values
        set_fixed_seeds(42)
        assert random.random() == val_py
        assert np.random.rand() == val_np

    def test_compute_dataset_fingerprint(self, tmp_path: Path):
        f = tmp_path / "corpus.jsonl"
        f.write_text("sample content", encoding="utf-8")
        fp = compute_dataset_fingerprint(tmp_path)
        assert "corpus.jsonl" in fp
        assert len(fp["corpus.jsonl"]) == 64  # SHA256 length


class TestRunResumptionAndSafety:
    def test_load_completed_run_keys(self, tmp_path: Path):
        runs_file = tmp_path / "runs.jsonl"
        records = [
            {"query_id": "q1", "method": "bm25", "K": 5},
            {"query_id": "q1", "method": "bm25", "K": 10},
            {"query_id": "q2", "method": "bm25", "K": 5},
        ]
        with open(runs_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        completed = load_completed_run_keys(runs_file)
        assert len(completed) == 3
        assert ("q1", "bm25", 5) in completed
        assert ("q1", "bm25", 10) in completed
        assert ("q2", "bm25", 5) in completed
        assert ("q2", "bm25", 10) not in completed


class TestPipelineExecutionWithBM25:
    """Test experiment runner with BM25 on a small query limit."""

    def test_run_experiment_pipeline_bm25(self, tmp_path: Path, monkeypatch):
        # Override output paths to temp directory to avoid touching real experiment results
        runs_path = tmp_path / "test_runs.jsonl"
        per_query_path = tmp_path / "test_per_query.csv"
        summary_path = tmp_path / "test_summary.csv"
        meta_path = tmp_path / "test_meta.json"

        monkeypatch.setattr("src.experiments.run_experiments.RAW_RUNS_PATH", runs_path)
        monkeypatch.setattr("src.experiments.run_experiments.PER_QUERY_CSV_PATH", per_query_path)
        monkeypatch.setattr("src.experiments.run_experiments.SUMMARY_CSV_PATH", summary_path)
        monkeypatch.setattr("src.experiments.run_experiments.RUN_METADATA_PATH", meta_path)

        df_per_query, df_summary = run_experiment_pipeline(
            dataset_name="scifact",
            k_values=[5, 10],
            methods=["bm25"],
            seed=42,
            overwrite=True,
            limit=5,  # 5 test queries for fast verification
        )

        # 1. Verify scifact_runs.jsonl
        assert runs_path.exists()
        records = []
        with open(runs_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        # 5 queries * 2 K values = 10 records
        assert len(records) == 10
        for r in records:
            assert r["dataset"] == "scifact"
            assert "query_id" in r
            assert r["method"] == "bm25"
            assert r["K"] in [5, 10]
            assert "retrieved_document_ids" in r
            assert len(r["retrieved_document_ids"]) == r["K"]
            assert "ranked_scores" in r
            assert len(r["ranked_scores"]) == r["K"]
            assert "latency_ms" in r
            assert "relevant_document_ids" in r
            assert "evidence_units_retrieved" in r
            assert "query_level_metrics" in r
            metrics = r["query_level_metrics"]
            assert f"Recall@{r['K']}" in metrics
            assert f"Precision@{r['K']}" in metrics
            assert "MRR" in metrics

        # 2. Verify per_query CSV
        assert per_query_path.exists()
        assert len(df_per_query) == 10
        expected_per_query_cols = [
            "dataset",
            "query_id",
            "method",
            "K",
            "latency_ms",
            "num_retrieved",
            "num_relevant",
            "num_relevant_retrieved",
            "num_evidence_annotated",
            "num_evidence_retrieved",
            "Recall@K",
            "Precision@K",
            "MRR",
            "EvidenceRecall@K",
            "EvidenceCompleteness@K",
        ]
        for col in expected_per_query_cols:
            assert col in df_per_query.columns

        # 3. Verify summary CSV
        assert summary_path.exists()
        assert len(df_summary) == 2  # BM25 K=5 and BM25 K=10
        expected_summary_cols = [
            "dataset",
            "method",
            "K",
            "query_count",
            "Recall@K",
            "Precision@K",
            "MRR",
            "Evidence Recall@K",
            "Evidence Completeness@K",
            "mean latency",
            "median latency",
        ]
        for col in expected_summary_cols:
            assert col in df_summary.columns

        # Summary values sanity checks
        for _, row in df_summary.iterrows():
            assert row["dataset"] == "scifact"
            assert row["method"] == "bm25"
            assert row["query_count"] == 5
            assert 0.0 <= row["Recall@K"] <= 1.0
            assert 0.0 <= row["Precision@K"] <= 1.0
            assert 0.0 <= row["MRR"] <= 1.0
            assert row["mean latency"] >= 0.0

        # 4. Verify metadata JSON
        assert meta_path.exists()
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["seed"] == 42
        assert "run_timestamp" in meta
        assert "retrieval_configuration" in meta
        assert "environment" in meta
        assert "dataset_fingerprint" in meta

    def test_run_experiment_pipeline_resumes_without_duplication(self, tmp_path: Path, monkeypatch):
        runs_path = tmp_path / "test_runs.jsonl"
        per_query_path = tmp_path / "test_per_query.csv"
        summary_path = tmp_path / "test_summary.csv"
        meta_path = tmp_path / "test_meta.json"

        monkeypatch.setattr("src.experiments.run_experiments.RAW_RUNS_PATH", runs_path)
        monkeypatch.setattr("src.experiments.run_experiments.PER_QUERY_CSV_PATH", per_query_path)
        monkeypatch.setattr("src.experiments.run_experiments.SUMMARY_CSV_PATH", summary_path)
        monkeypatch.setattr("src.experiments.run_experiments.RUN_METADATA_PATH", meta_path)

        # First run: limit=3 queries
        df_pq1, df_s1 = run_experiment_pipeline(
            dataset_name="scifact",
            k_values=[5],
            methods=["bm25"],
            overwrite=True,
            limit=3,
        )
        assert len(df_pq1) == 3

        # Second run: limit=5 queries without overwrite -> should resume and only add 2 more queries
        df_pq2, df_s2 = run_experiment_pipeline(
            dataset_name="scifact",
            k_values=[5],
            methods=["bm25"],
            overwrite=False,
            limit=5,
        )
        assert len(df_pq2) == 5
