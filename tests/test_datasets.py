"""Unit tests for SciFact dataset parsing, validation, and metadata extraction."""

import json
from pathlib import Path
import pytest

from src.datasets.scifact import (
    SciFactDocument,
    SciFactQuery,
    extract_evidence_from_metadata,
    load_scifact_dataset,
    parse_corpus,
    parse_queries,
    parse_qrels,
)
from src.datasets.inspect_dataset import inspect_scifact
from src.evaluation.metrics import EvidenceItem


class TestCorpusParsing:
    def test_parse_corpus_valid(self, tmp_path: Path):
        corpus_file = tmp_path / "corpus.jsonl"
        lines = [
            json.dumps({"_id": "doc1", "title": "Title 1", "text": "Body text 1", "metadata": {"year": 2020}}),
            json.dumps({"_id": "doc2", "title": "Title 2", "text": "Body text 2"}),
        ]
        corpus_file.write_text("\n".join(lines), encoding="utf-8")

        corpus = parse_corpus(corpus_file)
        assert len(corpus) == 2
        assert "doc1" in corpus and "doc2" in corpus

        doc1 = corpus["doc1"]
        assert doc1.id == "doc1"
        assert doc1.title == "Title 1"
        assert doc1.text == "Body text 1"
        assert doc1.full_text == "Title 1\nBody text 1"
        assert doc1.metadata == {"year": 2020}

    def test_parse_corpus_duplicate_doc_id_raises(self, tmp_path: Path):
        corpus_file = tmp_path / "corpus.jsonl"
        lines = [
            json.dumps({"_id": "doc1", "title": "First", "text": "A"}),
            json.dumps({"_id": "doc1", "title": "Second", "text": "B"}),
        ]
        corpus_file.write_text("\n".join(lines), encoding="utf-8")

        with pytest.raises(ValueError, match="Duplicate doc_id 'doc1'"):
            parse_corpus(corpus_file)


class TestQueriesParsing:
    def test_extract_evidence_from_metadata(self):
        metadata = {
            "docA": [{"sentences": [1, 3], "label": "SUPPORT"}],
            "docB": [{"sentences": [0], "label": "CONTRADICT"}],
        }
        evidence = extract_evidence_from_metadata(metadata)
        assert len(evidence) == 3
        assert EvidenceItem("docA", 1) in evidence
        assert EvidenceItem("docA", 3) in evidence
        assert EvidenceItem("docB", 0) in evidence

    def test_parse_queries_valid(self, tmp_path: Path):
        queries_file = tmp_path / "queries.jsonl"
        lines = [
            json.dumps({
                "_id": "q1",
                "text": "Claim 1",
                "metadata": {"doc1": [{"sentences": [2], "label": "SUPPORT"}]},
            }),
            json.dumps({"_id": "q2", "text": "Claim 2", "metadata": {}}),
        ]
        queries_file.write_text("\n".join(lines), encoding="utf-8")

        queries = parse_queries(queries_file)
        assert len(queries) == 2
        assert queries["q1"].id == "q1"
        assert queries["q1"].text == "Claim 1"
        assert len(queries["q1"].evidence) == 1
        assert queries["q1"].evidence[0] == EvidenceItem("doc1", 2)

        assert queries["q2"].id == "q2"
        assert queries["q2"].evidence == []

    def test_parse_queries_duplicate_id_raises(self, tmp_path: Path):
        queries_file = tmp_path / "queries.jsonl"
        lines = [
            json.dumps({"_id": "q1", "text": "Claim A"}),
            json.dumps({"_id": "q1", "text": "Claim B"}),
        ]
        queries_file.write_text("\n".join(lines), encoding="utf-8")

        with pytest.raises(ValueError, match="Duplicate query_id 'q1'"):
            parse_queries(queries_file)


class TestQrelsParsing:
    def test_parse_qrels_valid(self, tmp_path: Path):
        qrels_file = tmp_path / "test.tsv"
        content = "query-id\tcorpus-id\tscore\nq1\tdoc1\t1\nq1\tdoc2\t1\nq2\tdoc3\t1\n"
        qrels_file.write_text(content, encoding="utf-8")

        qrels = parse_qrels(qrels_file)
        assert len(qrels) == 2
        assert qrels["q1"] == {"doc1": 1, "doc2": 1}
        assert qrels["q2"] == {"doc3": 1}

    def test_parse_qrels_malformed_header_raises(self, tmp_path: Path):
        qrels_file = tmp_path / "test.tsv"
        qrels_file.write_text("invalid_header\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Malformed or empty qrels file"):
            parse_qrels(qrels_file)

    def test_parse_qrels_empty_ids_raises(self, tmp_path: Path):
        qrels_file = tmp_path / "test.tsv"
        content = "query-id\tcorpus-id\tscore\n\tdoc1\t1\n"
        qrels_file.write_text(content, encoding="utf-8")

        with pytest.raises(ValueError, match="Empty qid or docid"):
            parse_qrels(qrels_file)


class TestDatasetInspection:
    def test_inspect_dataset_synthetic(self, tmp_path: Path):
        scifact_dir = tmp_path / "scifact"
        scifact_dir.mkdir()
        qrels_dir = scifact_dir / "qrels"
        qrels_dir.mkdir()

        # Write mock corpus (including one empty doc)
        corpus_data = [
            {"_id": "doc1", "title": "Title 1", "text": "This is a five word text."},
            {"_id": "doc2", "title": "", "text": ""},  # empty doc
        ]
        (scifact_dir / "corpus.jsonl").write_text(
            "\n".join(json.dumps(d) for d in corpus_data), encoding="utf-8"
        )

        # Write mock queries
        queries_data = [
            {"_id": "q1", "text": "Claim 1", "metadata": {"doc1": [{"sentences": [0], "label": "SUPPORT"}]}},
        ]
        (scifact_dir / "queries.jsonl").write_text(
            "\n".join(json.dumps(q) for q in queries_data), encoding="utf-8"
        )

        # Write mock qrels referencing doc1 and a non-existent docX
        qrels_content = "query-id\tcorpus-id\tscore\nq1\tdoc1\t1\nq1\tdocX\t1\n"
        (qrels_dir / "test.tsv").write_text(qrels_content, encoding="utf-8")

        report = inspect_scifact(scifact_dir, split="test")

        assert report["counts"]["corpus_count"] == 2
        assert report["counts"]["query_count"] == 1
        assert report["integrity"]["empty_documents_count"] == 1
        assert report["integrity"]["empty_documents"] == ["doc2"]
        assert report["integrity"]["missing_docs_in_qrels_count"] == 1
        assert report["integrity"]["missing_docs_in_qrels"] == ["docX"]
        assert report["evidence_annotations"]["all_queries_with_evidence"] == 1


class TestRealSciFactIntegration:
    @pytest.mark.skipif(
        not Path("data/raw/scifact/corpus.jsonl").exists(),
        reason="SciFact raw dataset not present",
    )
    def test_load_real_scifact(self):
        dataset = load_scifact_dataset("data/raw/scifact", split="test")
        assert len(dataset.corpus) == 5183
        assert len(dataset.queries) == 1109
        assert len(dataset.qrels) == 300
        # Verify that original IDs and metadata are preserved
        first_doc = next(iter(dataset.corpus.values()))
        assert first_doc.id != ""
        assert first_doc.text != ""
