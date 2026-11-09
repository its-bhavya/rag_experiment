"""Unit tests for retrieval and evidence completeness metrics.

Includes:
- Document-level metrics: Recall@K, Precision@K, MRR
- Evidence-level metrics: Evidence Recall@K, Evidence Completeness@K
- Passage-to-document mapping & preservation of doc-to-passage links
- Context-sensitive evidence matching (a retrieved document does NOT imply its
  evidence sentences were retrieved unless the retrieval context includes them)
- Exact and normalized text matching (unicode, dashes, curly quotes)
- Hand-constructed manual calculation examples across multiple queries
- Edge cases:
  - No relevant documents
  - Unmatched evidence
  - Empty result lists
  - Duplicate retrieved IDs
  - Fewer than K documents retrieved
  - Configurable K values
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    EvidenceItem,
    QueryEvaluation,
    RetrievedPassage,
    aggregate_evaluations,
    evaluate_query,
    evidence_completeness_at_k,
    evidence_recall_at_k,
    get_covered_evidence,
    is_evidence_covered_by_passage,
    map_passages_to_documents,
    matches_evidence_text,
    normalize_evidence_text,
    precision_at_k,
    preserve_doc_to_passages,
    recall_at_k,
    reciprocal_rank,
)


class TestDocumentLevelMetrics:
    def test_recall_at_k_perfect(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc2"}
        assert recall_at_k(retrieved, relevant, k=2) == 1.0
        assert recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_recall_at_k_partial(self):
        retrieved = ["doc1", "docX", "docY", "docZ", "doc2"]
        relevant = {"doc1", "doc2"}
        # At k=3, only doc1 is retrieved (1/2 = 0.5)
        assert recall_at_k(retrieved, relevant, k=3) == 0.5
        # At k=5, both doc1 and doc2 are retrieved (2/2 = 1.0)
        assert recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_recall_at_k_zero(self):
        retrieved = ["docA", "docB", "docC"]
        relevant = {"doc1", "doc2"}
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_recall_at_k_empty_retrieved(self):
        retrieved = []
        relevant = {"doc1"}
        assert recall_at_k(retrieved, relevant, k=5) == 0.0

    def test_precision_at_k(self):
        retrieved = ["doc1", "docX", "doc2", "docY", "docZ"]
        relevant = {"doc1", "doc2"}
        # Top 5 contains 2 relevant: 2/5 = 0.4
        assert precision_at_k(retrieved, relevant, k=5) == 0.4
        # Top 1 contains 1 relevant: 1/1 = 1.0
        assert precision_at_k(retrieved, relevant, k=1) == 1.0
        # Top 2 contains 1 relevant: 1/2 = 0.5
        assert precision_at_k(retrieved, relevant, k=2) == 0.5

    def test_precision_at_k_shorter_than_k(self):
        # When retrieved list is shorter than k, denominator is still k
        retrieved = ["doc1"]
        relevant = {"doc1", "doc2"}
        assert precision_at_k(retrieved, relevant, k=5) == 0.2

    def test_reciprocal_rank(self):
        retrieved = ["docX", "docY", "doc1", "doc2"]
        relevant = {"doc1", "doc2"}
        # First relevant doc is doc1 at rank 3: RR = 1/3
        assert np.isclose(reciprocal_rank(retrieved, relevant), 1.0 / 3.0)

    def test_reciprocal_rank_with_cutoff(self):
        retrieved = ["docX", "docY", "doc1"]
        relevant = {"doc1"}
        # With cutoff=2, doc1 at rank 3 is ignored
        assert reciprocal_rank(retrieved, relevant, cutoff=2) == 0.0
        # With cutoff=3, doc1 is included
        assert np.isclose(reciprocal_rank(retrieved, relevant, cutoff=3), 1.0 / 3.0)

    def test_reciprocal_rank_no_hit(self):
        retrieved = ["docX", "docY"]
        relevant = {"doc1"}
        assert reciprocal_rank(retrieved, relevant) == 0.0
        assert reciprocal_rank([], relevant) == 0.0


class TestEvidenceMetrics:
    def test_evidence_item_creation(self):
        e = EvidenceItem(doc_id="doc1", evidence_id=3)
        assert e.doc_id == "doc1"
        assert e.evidence_id == 3
        assert e.text is None

        e_with_text = EvidenceItem(doc_id="doc1", evidence_id="sent_1", text="Sample evidence.")
        assert e_with_text.text == "Sample evidence."

        with pytest.raises(ValueError, match="non-empty string"):
            EvidenceItem(doc_id="   ", evidence_id=1)

    def test_evidence_recall_multi_doc(self):
        # Multi-hop: query requires 2 evidence pieces from 2 distinct documents
        evidence = [
            EvidenceItem("doc1", "sent_0"),
            EvidenceItem("doc2", "sent_4"),
        ]
        retrieved = ["doc1", "docX", "docY"]
        # Only doc1 is retrieved in top-3
        assert evidence_recall_at_k(retrieved, evidence, k=3) == 0.5

        # Both doc1 and doc2 retrieved in top-5
        retrieved_5 = ["doc1", "docX", "docY", "doc2", "docZ"]
        assert evidence_recall_at_k(retrieved_5, evidence, k=5) == 1.0

    def test_evidence_recall_same_doc_multiple_facts(self):
        # Claim requires 2 sentences from the same document
        evidence = [
            EvidenceItem("doc1", 0),
            EvidenceItem("doc1", 2),
        ]
        # If doc1 is in top-k as full document, both evidence items are retrieved
        retrieved = ["doc1", "docX"]
        assert evidence_recall_at_k(retrieved, evidence, k=2) == 1.0

        # If doc1 is not retrieved, evidence recall is 0.0
        retrieved_other = ["docA", "docB"]
        assert evidence_recall_at_k(retrieved_other, evidence, k=2) == 0.0

    def test_evidence_completeness_critical_discrepancy(self):
        """Validates the core research premise:

        Document-level Recall can be non-zero (0.5), yet Evidence Completeness is 0.0!
        """
        evidence = [
            EvidenceItem("docA", 1),
            EvidenceItem("docB", 2),
        ]
        relevant_docs = {"docA", "docB"}

        # System retrieves docA at rank 1, but docB is missing from top-5
        retrieved = ["docA", "other1", "other2", "other3", "other4"]

        # Document Recall@5 = 1/2 = 0.5
        assert recall_at_k(retrieved, relevant_docs, k=5) == 0.5
        # Evidence Recall@5 = 1/2 = 0.5
        assert evidence_recall_at_k(retrieved, evidence, k=5) == 0.5
        # Evidence Completeness@5 = 0.0 (question cannot be fully answered)
        assert evidence_completeness_at_k(retrieved, evidence, k=5) == 0.0

        # Now system retrieves docB at rank 6: top-5 is still incomplete, top-10 is complete
        retrieved_10 = ["docA", "other1", "other2", "other3", "other4", "docB"]
        assert evidence_completeness_at_k(retrieved_10, evidence, k=5) == 0.0
        assert evidence_completeness_at_k(retrieved_10, evidence, k=10) == 1.0


class TestPassageRetrievalAndEvidenceCoverage:
    """Tests requirement:

    A document being retrieved must NOT automatically imply all of its supporting
    sentences were retrieved unless the retrieval context actually includes those sentences.
    """

    def test_passage_matches_parent_doc_but_missing_sentence(self):
        # Gold evidence is in docA, sentence "Target therapy induces tumor apoptosis."
        evidence = [
            EvidenceItem(
                doc_id="docA",
                evidence_id="sent_target",
                text="Target therapy induces tumor apoptosis.",
            )
        ]

        # Retriever retrieved a passage from docA, but paragraph 3 which talks about something else!
        passage_irrelevant = RetrievedPassage(
            doc_id="docA",
            passage_id="p3",
            text="Patient cohort demographics included 50 male and 40 female participants.",
            rank=1,
            is_full_document=False,
        )

        assert not is_evidence_covered_by_passage(evidence[0], passage_irrelevant)
        assert evidence_recall_at_k([passage_irrelevant], evidence, k=1) == 0.0
        assert evidence_completeness_at_k([passage_irrelevant], evidence, k=1) == 0.0

    def test_passage_matches_parent_doc_and_contains_sentence(self):
        evidence = [
            EvidenceItem(
                doc_id="docA",
                evidence_id="sent_target",
                text="Target therapy induces tumor apoptosis.",
            )
        ]

        passage_relevant = RetrievedPassage(
            doc_id="docA",
            passage_id="p1",
            text="In vitro assays demonstrated that target therapy induces tumor apoptosis significantly.",
            rank=1,
            is_full_document=False,
        )

        assert is_evidence_covered_by_passage(evidence[0], passage_relevant)
        assert evidence_recall_at_k([passage_relevant], evidence, k=1) == 1.0
        assert evidence_completeness_at_k([passage_relevant], evidence, k=1) == 1.0

    def test_normalized_text_matching_with_punctuation_and_unicode(self):
        # Evidence with curly quotes, unicode minus/dash, extra spacing
        ev_text = "IL-6 inhibits T-cell activation."
        ctx_text = "Previous studies revealed that IL—6  inhibits ‘T-cell’ activation."

        assert not (ev_text in ctx_text)  # Raw substring fails due to em-dash and curly quotes
        assert matches_evidence_text(ev_text, ctx_text)  # Normalized matching succeeds

        evidence = EvidenceItem(doc_id="doc1", evidence_id=0, text=ev_text)
        passage = RetrievedPassage(doc_id="doc1", text=ctx_text, rank=1)
        assert is_evidence_covered_by_passage(evidence, passage)

    def test_full_document_flag_behavior(self):
        evidence = EvidenceItem(doc_id="doc1", evidence_id=0, text="Important fact.")

        # Full document without text breakdown represents full document retrieval
        full_doc = RetrievedPassage(doc_id="doc1", is_full_document=True, rank=1)
        assert is_evidence_covered_by_passage(evidence, full_doc)

        # Full document WITH text checks text inclusion if text is present
        full_doc_with_mismatch = RetrievedPassage(
            doc_id="doc1",
            text="Completely different abstract text.",
            is_full_document=True,
            rank=1,
        )
        assert not is_evidence_covered_by_passage(evidence, full_doc_with_mismatch)


class TestPassageToDocumentMapping:
    def test_map_passages_to_documents_deduplicates_preserving_first_rank(self):
        passages = [
            RetrievedPassage(doc_id="docA", passage_id="p1", rank=1),
            RetrievedPassage(doc_id="docB", passage_id="p1", rank=2),
            RetrievedPassage(doc_id="docA", passage_id="p2", rank=3),  # duplicate docA
            RetrievedPassage(doc_id="docC", passage_id="p1", rank=4),
            RetrievedPassage(doc_id="docB", passage_id="p2", rank=5),  # duplicate docB
        ]

        doc_ids = map_passages_to_documents(passages)
        assert doc_ids == ["docA", "docB", "docC"]

    def test_preserve_doc_to_passages(self):
        passages = [
            RetrievedPassage(doc_id="docA", passage_id="p1", rank=1),
            RetrievedPassage(doc_id="docB", passage_id="p1", rank=2),
            RetrievedPassage(doc_id="docA", passage_id="p2", rank=3),
        ]
        doc_map = preserve_doc_to_passages(passages)
        assert len(doc_map["docA"]) == 2
        assert [p.passage_id for p in doc_map["docA"]] == ["p1", "p2"]
        assert len(doc_map["docB"]) == 1


class TestHandConstructedExamples:
    """Hand-constructed multi-query evaluation with manually verified ground truth."""

    def test_hand_calculated_multi_query_dataset(self):
        # Query 1: Requires 2 documents {d1, d2} and 2 evidence items {e1 in d1, e2 in d2}
        # Retriever retrieves: [d1, d_irrel, d2, d_other]
        # Cutoffs: K in (1, 2, 3)
        # Expected Q1:
        #   Recall@1 = 1/2 = 0.5
        #   Recall@2 = 1/2 = 0.5
        #   Recall@3 = 2/2 = 1.0
        #   Precision@1 = 1/1 = 1.0
        #   Precision@2 = 1/2 = 0.5
        #   Precision@3 = 2/3 = 0.6666667
        #   MRR = 1/1 = 1.0
        #   EvidenceRecall@1 = 1/2 = 0.5
        #   EvidenceRecall@2 = 1/2 = 0.5
        #   EvidenceRecall@3 = 2/2 = 1.0
        #   EvidenceCompleteness@1 = 0.0
        #   EvidenceCompleteness@2 = 0.0
        #   EvidenceCompleteness@3 = 1.0
        q1_relevant = {"d1", "d2"}
        q1_evidence = [EvidenceItem("d1", 0), EvidenceItem("d2", 1)]
        q1_retrieved = ["d1", "d_irrel", "d2", "d_other"]

        ev1 = evaluate_query("q1", q1_retrieved, q1_relevant, q1_evidence, cutoffs=(1, 2, 3))
        assert np.isclose(ev1.metrics["Recall@1"], 0.5)
        assert np.isclose(ev1.metrics["Recall@2"], 0.5)
        assert np.isclose(ev1.metrics["Recall@3"], 1.0)
        assert np.isclose(ev1.metrics["Precision@1"], 1.0)
        assert np.isclose(ev1.metrics["Precision@2"], 0.5)
        assert np.isclose(ev1.metrics["Precision@3"], 2.0 / 3.0)
        assert np.isclose(ev1.metrics["MRR"], 1.0)
        assert np.isclose(ev1.metrics["EvidenceRecall@1"], 0.5)
        assert np.isclose(ev1.metrics["EvidenceRecall@3"], 1.0)
        assert np.isclose(ev1.metrics["EvidenceCompleteness@1"], 0.0)
        assert np.isclose(ev1.metrics["EvidenceCompleteness@3"], 1.0)

        # Query 2: Requires 1 document {d3} and 1 evidence item {e3 in d3}
        # Retriever retrieves: [d_miss, d3]
        # Expected Q2:
        #   Recall@1 = 0/1 = 0.0
        #   Recall@2 = 1/1 = 1.0
        #   Precision@1 = 0/1 = 0.0
        #   Precision@2 = 1/2 = 0.5
        #   MRR = 1/2 = 0.5
        #   EvidenceRecall@1 = 0.0
        #   EvidenceRecall@2 = 1.0
        #   EvidenceCompleteness@1 = 0.0
        #   EvidenceCompleteness@2 = 1.0
        q2_relevant = {"d3"}
        q2_evidence = [EvidenceItem("d3", 0)]
        q2_retrieved = ["d_miss", "d3"]

        ev2 = evaluate_query("q2", q2_retrieved, q2_relevant, q2_evidence, cutoffs=(1, 2, 3))
        assert np.isclose(ev2.metrics["Recall@1"], 0.0)
        assert np.isclose(ev2.metrics["Recall@2"], 1.0)
        assert np.isclose(ev2.metrics["Precision@1"], 0.0)
        assert np.isclose(ev2.metrics["Precision@2"], 0.5)
        assert np.isclose(ev2.metrics["MRR"], 0.5)
        assert np.isclose(ev2.metrics["EvidenceCompleteness@2"], 1.0)

        # Aggregate across [q1, q2]:
        # Macro MRR = (1.0 + 0.5) / 2 = 0.75
        # Macro Recall@1 = (0.5 + 0.0) / 2 = 0.25
        # Macro Recall@2 = (0.5 + 1.0) / 2 = 0.75
        # Macro Precision@1 = (1.0 + 0.0) / 2 = 0.5
        # Macro Precision@2 = (0.5 + 0.5) / 2 = 0.5
        # Macro EvidenceCompleteness@1 = (0.0 + 0.0) / 2 = 0.0
        # Macro EvidenceCompleteness@2 = (0.0 + 1.0) / 2 = 0.5
        # Macro EvidenceCompleteness@3 = (1.0 + 1.0) / 2 = 1.0
        agg = aggregate_evaluations([ev1, ev2])
        assert np.isclose(agg["MRR"], 0.75)
        assert np.isclose(agg["Recall@1"], 0.25)
        assert np.isclose(agg["Recall@2"], 0.75)
        assert np.isclose(agg["Precision@1"], 0.5)
        assert np.isclose(agg["Precision@2"], 0.5)
        assert np.isclose(agg["EvidenceCompleteness@1"], 0.0)
        assert np.isclose(agg["EvidenceCompleteness@2"], 0.5)
        assert np.isclose(agg["EvidenceCompleteness@3"], 1.0)


class TestEdgeCasesAndValidation:
    def test_no_relevant_documents_in_results(self):
        # Query has relevant documents, but retriever found none
        retrieved = ["doc_irrel1", "doc_irrel2"]
        relevant = {"doc_gold"}
        assert recall_at_k(retrieved, relevant, k=2) == 0.0
        assert precision_at_k(retrieved, relevant, k=2) == 0.0
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_no_relevant_documents_in_ground_truth(self):
        # Query has no relevant documents in ground truth
        retrieved = ["doc1", "doc2"]
        # With empty_relevant_handling="zero": returns 0.0
        assert recall_at_k(retrieved, set(), k=2, empty_relevant_handling="zero") == 0.0
        assert precision_at_k(retrieved, set(), k=2, empty_relevant_handling="zero") == 0.0
        assert reciprocal_rank(retrieved, set(), empty_relevant_handling="zero") == 0.0

        # With default "raise": raises ValueError
        with pytest.raises(ValueError, match="relevant_docs cannot be empty"):
            recall_at_k(retrieved, set(), k=2)

    def test_empty_retrieved_list(self):
        relevant = {"doc1"}
        evidence = [EvidenceItem("doc1", 0)]
        assert recall_at_k([], relevant, k=5) == 0.0
        assert precision_at_k([], relevant, k=5) == 0.0
        assert reciprocal_rank([], relevant) == 0.0
        assert evidence_recall_at_k([], evidence, k=5) == 0.0
        assert evidence_completeness_at_k([], evidence, k=5) == 0.0

    def test_unmatched_evidence(self):
        # Retrieved documents do not match any evidence
        retrieved = ["docX", "docY"]
        evidence = [EvidenceItem("docZ", 1)]
        assert evidence_recall_at_k(retrieved, evidence, k=5) == 0.0
        assert evidence_completeness_at_k(retrieved, evidence, k=5) == 0.0

    def test_duplicate_retrieved_ids_in_doc_ranking_rejected(self):
        corrupted = ["doc1", "doc2", "doc1"]
        relevant = {"doc1"}
        with pytest.raises(ValueError, match="duplicate doc_id"):
            recall_at_k(corrupted, relevant, k=5, allow_duplicates=False)

    def test_fewer_than_k_documents_precision_denominator_is_k(self):
        # Only 2 documents returned when K=10; both are relevant
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1", "doc2"}
        # Precision@10 must be 2 / 10 = 0.2 (NOT 2 / 2 = 1.0)
        assert precision_at_k(retrieved, relevant, k=10) == 0.2

    def test_invalid_k_rejected(self):
        retrieved = ["doc1"]
        relevant = {"doc1"}
        evidence = [EvidenceItem("doc1", 0)]

        with pytest.raises(ValueError, match="positive integer"):
            recall_at_k(retrieved, relevant, k=0)
        with pytest.raises(ValueError, match="positive integer"):
            precision_at_k(retrieved, relevant, k=-1)
        with pytest.raises(ValueError, match="positive integer"):
            evidence_recall_at_k(retrieved, evidence, k=0)
        with pytest.raises(ValueError, match="positive integer"):
            evidence_completeness_at_k(retrieved, evidence, k=-5)

    def test_blank_doc_id_rejected(self):
        corrupted = ["doc1", "   ", "doc2"]
        relevant = {"doc1"}
        with pytest.raises(ValueError, match="empty doc_id"):
            recall_at_k(corrupted, relevant, k=5)

    def test_aggregate_empty_raises(self):
        with pytest.raises(ValueError, match="Cannot aggregate empty"):
            aggregate_evaluations([])
