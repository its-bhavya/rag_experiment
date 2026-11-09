"""Unit tests for SciFact evidence mapping, normalization, and matching."""

import json
from pathlib import Path
import pytest

from src.datasets.evidence_mapper import (
    EvidenceMappingRecord,
    build_evidence_mappings,
    clean_tokens,
    match_sentence_in_document,
    normalize_for_matching,
)


class TestTextNormalization:
    def test_unicode_and_smart_quotes(self):
        text = "“Smart double quotes” and ‘smart single quotes’"
        norm = normalize_for_matching(text)
        assert norm == '"Smart double quotes" and \'smart single quotes\''

    def test_varied_dashes_and_hyphens(self):
        # En-dash (\u2013), em-dash (\u2014), minus sign (\u2212)
        text = "COVID–19 is a pandemic—global scale −5 degrees"
        norm = normalize_for_matching(text)
        assert norm == "COVID-19 is a pandemic-global scale -5 degrees"

    def test_whitespace_collapsing(self):
        text = "Sentence   with \n multiple \t\t spaces   and newlines.  \n"
        norm = normalize_for_matching(text)
        assert norm == "Sentence with multiple spaces and newlines."

    def test_clean_tokens(self):
        tokens = clean_tokens("Alpha(+)-thalassaemia reduces malaria-risk (p = 0.09)!")
        assert "alpha" in tokens
        assert "thalassaemia" in tokens
        assert "reduces" in tokens
        assert "malaria" in tokens
        assert "risk" in tokens


class TestSentenceMatching:
    def test_exact_literal_match(self):
        sentence = "This is an exact supporting evidence sentence."
        doc = "Intro. This is an exact supporting evidence sentence. Conclusion."
        matched, method, conf, start, end, reason = match_sentence_in_document(sentence, doc)
        assert matched is True
        assert method == "exact"
        assert conf == 1.0
        assert start == 7
        assert end == 7 + len(sentence)
        assert reason is None

    def test_normalized_punctuation_and_whitespace_match(self):
        # Sentence has curly quotes, em-dash, and extra spaces
        sentence = "ACE  inhibitor\u2014induced   functional ARF is \u201ccritical\u201d in practice.  \n"
        # Document in BEIR has standard hyphen, straight quotes, single spaces
        doc = "Understanding ACE inhibitor-induced functional ARF is \"critical\" in practice."

        matched, method, conf, start, end, reason = match_sentence_in_document(sentence, doc)
        assert matched is True
        assert "normalized" in method
        assert conf >= 0.95
        assert reason is None

    def test_section_header_prefix_match(self):
        # In SciFact, sentences sometimes start with embedded section tags like "BACKGROUND "
        sentence = "BACKGROUND Alterations of cerebral white matter affect cortical development in infants."
        doc = "Alterations of cerebral white matter affect cortical development in infants."

        matched, method, conf, start, end, reason = match_sentence_in_document(sentence, doc)
        assert matched is True
        assert "normalized" in method
        assert conf >= 0.90

    def test_token_overlap_fuzzy_match(self):
        # One word differs due to spelling / OCR variant
        sentence = "Haematological profile in children reduces the risk of severe anaemia."
        doc = "Hematological profile in children reduces the risk of severe anaemia in malaria."

        matched, method, conf, start, end, reason = match_sentence_in_document(
            sentence, doc, min_overlap_threshold=0.85
        )
        assert matched is True
        assert method in ("token_overlap", "normalized")
        assert conf >= 0.85


class TestUnmatchedEvidence:
    def test_unmatched_completely_absent(self):
        sentence = "Quantum computing achieves exponential speedup over classical circuits."
        doc = "Biomaterials and cellular scaffolds enhance tissue regeneration in vivo."

        matched, method, conf, start, end, reason = match_sentence_in_document(sentence, doc)
        assert matched is False
        assert method == "unmatched"
        assert conf < 0.85
        assert "low_similarity" in reason

    def test_unmatched_empty_inputs(self):
        matched, method, conf, start, end, reason = match_sentence_in_document("   ", "Some text")
        assert matched is False
        assert reason == "empty_evidence_sentence"

        matched, method, conf, start, end, reason = match_sentence_in_document("Some sentence", "   ")
        assert matched is False
        assert reason == "empty_document_text"


class TestBuildEvidenceMappings:
    def test_synthetic_mapping_pipeline(self, tmp_path: Path):
        # 1. Mock official corpus
        official_corpus_file = tmp_path / "official_corpus.jsonl"
        official_corpus_data = [
            {"doc_id": 101, "abstract": ["Sentence 0 of doc 101.", "Sentence 1 of doc 101."]},
            {"doc_id": 102, "abstract": ["Sentence 0 of doc 102."]},
        ]
        official_corpus_file.write_text(
            "\n".join(json.dumps(d) for d in official_corpus_data), encoding="utf-8"
        )

        # 2. Mock BEIR corpus (doc 101 has text, doc 102 is missing to test missing doc error)
        beir_corpus_file = tmp_path / "beir_corpus.jsonl"
        beir_corpus_data = [
            {"_id": "101", "text": "Sentence 0 of doc 101. Sentence 1 of doc 101."},
        ]
        beir_corpus_file.write_text(
            "\n".join(json.dumps(d) for d in beir_corpus_data), encoding="utf-8"
        )

        # 3. Mock claims
        # q1: complete match (sentence 0 and 1 of doc 101)
        # q2: partial match (sentence 0 of doc 101 matched, sentence 0 of missing doc 102 unmatched)
        # q3: unmatched out of bounds index (sentence 99 of doc 101)
        claims_file = tmp_path / "claims.jsonl"
        claims_data = [
            {
                "id": 1,
                "claim": "Claim 1",
                "evidence": {"101": [{"sentences": [0, 1], "label": "SUPPORT"}]},
            },
            {
                "id": 2,
                "claim": "Claim 2",
                "evidence": {
                    "101": [{"sentences": [0], "label": "SUPPORT"}],
                    "102": [{"sentences": [0], "label": "SUPPORT"}],
                },
            },
            {
                "id": 3,
                "claim": "Claim 3",
                "evidence": {"101": [{"sentences": [99], "label": "CONTRADICT"}]},
            },
        ]
        claims_file.write_text(
            "\n".join(json.dumps(c) for c in claims_data), encoding="utf-8"
        )

        records, report = build_evidence_mappings(
            claims_file, official_corpus_file, beir_corpus_file
        )

        # q1: 2 sentences, both matched -> complete query
        # q2: 2 sentences, 1 matched, 1 missing from beir -> partial query
        # q3: 1 sentence out of bounds -> zero matched query
        assert report["mapping_summary"]["total_evidence_annotations"] == 5
        assert report["mapping_summary"]["successfully_mapped_evidence"] == 3
        assert report["mapping_summary"]["unmatched_evidence"] == 2

        qc = report["query_completeness"]
        assert qc["queries_with_complete_evidence_mappings"] == ["1"]
        assert qc["queries_with_partial_evidence_mappings"] == ["2"]
        assert qc["queries_with_zero_evidence_mappings"] == ["3"]

        # Check unmapped reasons
        unmapped = [r for r in records if r.evidence_match_status == "unmatched"]
        reasons = {u.unmapped_reason for u in unmapped}
        assert any("missing_from_beir_corpus" in r for r in reasons)
        assert any("exceeds_abstract_len" in r for r in reasons)


class TestRealSciFactProcessedArtifacts:
    @pytest.mark.skipif(
        not Path("data/processed/scifact_evidence.jsonl").exists(),
        reason="data/processed/scifact_evidence.jsonl not found",
    )
    def test_real_processed_evidence_artifacts(self):
        evidence_file = Path("data/processed/scifact_evidence.jsonl")
        report_file = Path("data/processed/evidence_mapping_report.json")

        assert evidence_file.exists()
        assert report_file.exists()

        records = [json.loads(line) for line in open(evidence_file, encoding="utf-8")]
        assert len(records) == 366

        # Ensure all required fields exist on every record
        required_fields = {
            "query_id",
            "evidence_doc_id",
            "sentence_idx",
            "evidence_sentence",
            "evidence_annotation_id",
            "evidence_match_status",
            "match_method",
            "match_confidence",
        }
        for r in records:
            assert required_fields.issubset(r.keys())
            assert r["evidence_match_status"] in ("matched", "unmatched")
            assert len(r["evidence_sentence"]) > 0

        with open(report_file, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert report["mapping_summary"]["total_evidence_annotations"] == 366
        assert report["mapping_summary"]["successfully_mapped_evidence"] == 366
        assert report["mapping_summary"]["unmatched_evidence"] == 0
        assert report["query_completeness"]["queries_with_complete_evidence_mappings_count"] == 188

    def test_load_verified_evidence_ground_truth(self):
        from src.datasets.evidence_mapper import load_verified_evidence_ground_truth

        evidence_by_query, eligible_qids = load_verified_evidence_ground_truth(
            "data/processed/scifact_evidence.jsonl",
            "data/processed/evidence_mapping_report.json",
        )
        assert len(eligible_qids) == 188
        assert len(evidence_by_query) == 188
        total_items = sum(len(items) for items in evidence_by_query.values())
        assert total_items == 366
        for qid in eligible_qids:
            assert len(evidence_by_query[qid]) >= 1
