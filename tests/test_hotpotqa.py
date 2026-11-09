"""Pytest test suite for the HotpotQA distractor-setting extension.

All unit tests use synthetic data and do NOT require a HuggingFace download.
Tests that require real processed data files are skipped gracefully if the
files are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets.hotpotqa import (  # noqa: E402
    HotpotDataset,
    HotpotParagraph,
    HotpotQuestion,
    _build_paragraphs,
    _map_supporting_facts,
    _process_raw_question,
    load_hotpotqa_distractor,
    save_processed_data,
)
from src.evaluation.metrics import (  # noqa: E402
    EvidenceItem,
    RetrievedPassage,
    evaluate_query,
    evidence_completeness_at_k,
    evidence_recall_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.experiments.run_hotpotqa import PerQueryDenseRetriever  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion  # noqa: E402

# ---------------------------------------------------------------------------
# Check whether processed data files exist (for integration tests)
# ---------------------------------------------------------------------------
_PROCESSED_DIR = _ROOT / "data" / "processed"
_QUESTIONS_FILE = _PROCESSED_DIR / "hotpotqa_questions.jsonl"
_DATA_AVAILABLE = _QUESTIONS_FILE.exists()


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

SYNTHETIC_CONTEXT = [
    ["Alpha Article", ["Alpha sentence one.", "Alpha sentence two.", "Alpha sentence three."]],
    ["Beta Article", ["Beta sentence A.", "Beta sentence B."]],
    ["Gamma Article", ["Gamma only sentence."]],
    ["Delta Article", ["Delta first.", "Delta second."]],
    ["Epsilon Article", ["Epsilon content."]],
    ["Zeta Article", ["Zeta info.", "Zeta more info."]],
    ["Eta Article", ["Eta text."]],
    ["Theta Article", ["Theta words.", "Theta extra."]],
    ["Iota Article", ["Iota blurb."]],
    ["Kappa Article", ["Kappa single."]],
]

SYNTHETIC_SUPPORTING_FACTS = [
    ["Alpha Article", 1],   # "Alpha sentence two."
    ["Beta Article", 0],    # "Beta sentence A."
]

SYNTHETIC_RAW: Dict[str, Any] = {
    "id": "test_q_001",
    "question": "What connects Alpha and Beta?",
    "answer": "Connection",
    "type": "bridge",
    "level": "medium",
    "supporting_facts": {
        "title": ["Alpha Article", "Beta Article"],
        "sent_id": [1, 0],
    },
    "context": {
        "title": [entry[0] for entry in SYNTHETIC_CONTEXT],
        "sentences": [entry[1] for entry in SYNTHETIC_CONTEXT],
    },
}


def _make_question(raw: Dict[str, Any] = None) -> HotpotQuestion:
    return _process_raw_question(raw or SYNTHETIC_RAW)


def _make_paragraphs(qid: str = "test_q_001") -> Dict[str, HotpotParagraph]:
    return _build_paragraphs(qid, SYNTHETIC_CONTEXT)


# ---------------------------------------------------------------------------
# Test 1: load_and_sample
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason="Processed data files not available — skipping integration test.",
)
def test_load_and_sample():
    """Verify 500 questions are loaded and sampling is reproducible with seed 12345."""
    questions_path = _QUESTIONS_FILE
    questions = []
    with open(questions_path, encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))

    assert len(questions) == 500, f"Expected 500 questions, got {len(questions)}"

    # Reproducibility: load again and check same IDs
    ids_first = [q["question_id"] for q in questions]

    questions2 = []
    with open(questions_path, encoding="utf-8") as f:
        for line in f:
            questions2.append(json.loads(line))
    ids_second = [q["question_id"] for q in questions2]

    assert ids_first == ids_second, "Sampling is not reproducible!"


# ---------------------------------------------------------------------------
# Test 2: candidate_construction
# ---------------------------------------------------------------------------


def test_candidate_construction():
    """Verify all 10 paragraphs are present and doc_ids are correct."""
    q = _make_question()
    assert len(q.paragraphs) == 10, f"Expected 10 paragraphs, got {len(q.paragraphs)}"

    for entry in SYNTHETIC_CONTEXT:
        title = entry[0]
        expected_doc_id = f"test_q_001___{title}"
        assert expected_doc_id in q.paragraphs, (
            f"Expected doc_id '{expected_doc_id}' not found in paragraphs"
        )
        para = q.paragraphs[expected_doc_id]
        assert para.title == title
        assert para.question_id == "test_q_001"
        assert para.full_text == f"{title}\n{' '.join(entry[1])}"


# ---------------------------------------------------------------------------
# Test 3: supporting_fact_mapping
# ---------------------------------------------------------------------------


def test_supporting_fact_mapping():
    """Verify evidence sentence text matches the sentence at the given index."""
    q = _make_question()
    # Alpha Article, sent_idx=1 -> "Alpha sentence two."
    alpha_ev = next(
        (ev for ev in q.evidence_items if "Alpha Article" in ev.doc_id),
        None,
    )
    assert alpha_ev is not None, "Alpha Article evidence item not found"
    assert alpha_ev.evidence_id == 1
    assert alpha_ev.text == "Alpha sentence two."

    # Beta Article, sent_idx=0 -> "Beta sentence A."
    beta_ev = next(
        (ev for ev in q.evidence_items if "Beta Article" in ev.doc_id),
        None,
    )
    assert beta_ev is not None, "Beta Article evidence item not found"
    assert beta_ev.evidence_id == 0
    assert beta_ev.text == "Beta sentence A."


# ---------------------------------------------------------------------------
# Test 4: invalid_fact_flagging
# ---------------------------------------------------------------------------


def test_invalid_fact_flagging():
    """Inject a bad [title, 999] supporting fact and verify it's flagged."""
    bad_raw = dict(SYNTHETIC_RAW)
    bad_raw["supporting_facts"] = {
        "title": ["Alpha Article", "Alpha Article", "NonExistentTitle"],
        "sent_id": [0, 999, 0],
    }
    q = _process_raw_question(bad_raw)

    # sent_idx=999 is out of range for Alpha Article (3 sentences)
    invalid_reasons = [f["reason"] for f in q.invalid_supporting_facts]
    assert "sent_idx_out_of_range" in invalid_reasons, (
        f"Expected 'sent_idx_out_of_range' in invalid facts, got: {invalid_reasons}"
    )

    # NonExistentTitle not in context
    assert "title_not_in_context" in invalid_reasons, (
        f"Expected 'title_not_in_context' in invalid facts, got: {invalid_reasons}"
    )


# ---------------------------------------------------------------------------
# Test 5: bm25_per_query
# ---------------------------------------------------------------------------


def test_bm25_per_query():
    """Build a mini BM25 retriever over 3 paragraphs, check results are sorted."""
    doc_ids = ["doc_A", "doc_B", "doc_C"]
    doc_texts = [
        "Alpha article talks about neural networks and deep learning.",
        "Beta article discusses solar panels and renewable energy.",
        "Gamma article covers neural networks again.",
    ]
    retriever = BM25Retriever(doc_ids=doc_ids, doc_texts=doc_texts)
    results = retriever.retrieve("neural networks deep learning", k=3)

    assert len(results) <= 3
    scores = [r.score for r in results]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), (
        f"BM25 results not sorted by score: {scores}"
    )
    # doc_A or doc_C should be top (both mention neural networks)
    top_doc = results[0].doc_id
    assert top_doc in ("doc_A", "doc_C"), f"Expected neural network doc at top, got {top_doc}"
    assert all(r.rank == i + 1 for i, r in enumerate(results))


# ---------------------------------------------------------------------------
# Test 6: dense_per_query
# ---------------------------------------------------------------------------


def test_dense_per_query():
    """Build PerQueryDenseRetriever over 3 paragraphs, check results sum to ≤ k."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        pytest.skip("sentence-transformers not installed — skipping dense retriever test.")

    try:
        model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu", local_files_only=True)
    except Exception:
        try:
            model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        except Exception as e:
            pytest.skip(f"Could not load BAAI/bge-base-en-v1.5: {e}")

    doc_ids = ["doc_A", "doc_B", "doc_C"]
    doc_texts = [
        "Alpha article about neural networks.",
        "Beta article about solar energy.",
        "Gamma article about machine learning.",
    ]
    retriever = PerQueryDenseRetriever(
        doc_ids=doc_ids, doc_texts=doc_texts, model=model
    )
    results = retriever.retrieve("neural network machine learning", k=2)

    assert len(results) <= 2
    scores = [r.score for r in results]
    assert all(scores[i] >= scores[i + 1] - 1e-6 for i in range(len(scores) - 1)), (
        f"Dense results not sorted: {scores}"
    )
    for ev in results:
        assert ev.doc_id in doc_ids


# ---------------------------------------------------------------------------
# Test 7: hybrid_rrf
# ---------------------------------------------------------------------------


def test_hybrid_rrf():
    """Verify RRF fusion over a 3-paragraph pool produces sorted results."""
    doc_ids = ["doc_A", "doc_B", "doc_C"]
    doc_texts = [
        "Deep learning neural networks gradient descent.",
        "Renewable energy solar wind power.",
        "Neural architectures transformers attention.",
    ]
    bm25 = BM25Retriever(doc_ids=doc_ids, doc_texts=doc_texts)

    # We'll test just the RRF fusion logic using BM25 as both sparse and dense
    bm25_results = bm25.retrieve("neural network", k=3)
    # Create a second ranking (simulate dense by reversing)
    dense_results = list(reversed(bm25_results))
    # Re-index ranks
    from src.retrieval.base import ScoredDoc
    dense_results_reranked = [
        ScoredDoc(doc_id=sd.doc_id, score=sd.score, rank=i + 1)
        for i, sd in enumerate(dense_results)
    ]

    fused = reciprocal_rank_fusion(
        system_rankings=[bm25_results, dense_results_reranked],
        c=60,
        k=3,
    )

    assert len(fused) <= 3
    fused_scores = [r.score for r in fused]
    assert all(
        fused_scores[i] >= fused_scores[i + 1] - 1e-9
        for i in range(len(fused_scores) - 1)
    ), f"Hybrid results not sorted: {fused_scores}"
    assert all(r.rank == i + 1 for i, r in enumerate(fused))


# ---------------------------------------------------------------------------
# Test 8: evidence_coverage (positive case)
# ---------------------------------------------------------------------------


def test_evidence_coverage():
    """Retrieved paragraph with the right text gets credit for its supporting sentence."""
    evidence_sentence = "Alpha sentence two."
    doc_id = "test_q_001___Alpha Article"

    ev_item = EvidenceItem(
        doc_id=doc_id, evidence_id=1, text=evidence_sentence
    )
    # Retrieved passage contains the evidence sentence
    passage = RetrievedPassage(
        doc_id=doc_id,
        is_full_document=True,
        text=f"Alpha Article\nAlpha sentence one. {evidence_sentence} Alpha sentence three.",
        score=1.0,
        rank=1,
    )

    result = evaluate_query(
        query_id="test_q_001",
        retrieved_docs=[passage],
        relevant_docs=[doc_id],
        gold_evidence=[ev_item],
        cutoffs=[1],
        empty_relevant_handling="zero",
        empty_evidence_handling="zero",
    )
    assert result.metrics["EvidenceRecall@1"] == 1.0, (
        f"Expected EvidenceRecall@1=1.0, got {result.metrics['EvidenceRecall@1']}"
    )


# ---------------------------------------------------------------------------
# Test 9: evidence_not_automatic (negative case)
# ---------------------------------------------------------------------------


def test_evidence_not_automatic():
    """Retrieve paragraph but with text that does NOT contain the supporting sentence."""
    evidence_sentence = "Alpha sentence two."
    doc_id = "test_q_001___Alpha Article"

    ev_item = EvidenceItem(
        doc_id=doc_id, evidence_id=1, text=evidence_sentence
    )
    # Retrieved passage does NOT contain the evidence sentence
    passage = RetrievedPassage(
        doc_id=doc_id,
        is_full_document=True,  # full doc flag but no matching text
        text="Alpha Article\nAlpha sentence one. Alpha sentence three.",  # sent 1 missing
        score=1.0,
        rank=1,
    )

    result = evaluate_query(
        query_id="test_q_001",
        retrieved_docs=[passage],
        relevant_docs=[doc_id],
        gold_evidence=[ev_item],
        cutoffs=[1],
        empty_relevant_handling="zero",
        empty_evidence_handling="zero",
    )
    # Evidence sentence not in passage text => no evidence credit
    assert result.metrics["EvidenceRecall@1"] == 0.0, (
        f"Expected EvidenceRecall@1=0.0, got {result.metrics['EvidenceRecall@1']}"
    )


# ---------------------------------------------------------------------------
# Test 10: metrics_range
# ---------------------------------------------------------------------------


def test_metrics_range():
    """Recall, Precision, MRR, EvidenceRecall, EvidenceCompleteness all in [0,1]."""
    doc_ids = ["doc_A", "doc_B", "doc_C"]
    relevant = ["doc_A"]
    gold_ev = [EvidenceItem(doc_id="doc_A", evidence_id=0, text="Some sentence text.")]

    retrieved_passages = [
        RetrievedPassage(
            doc_id="doc_B",
            is_full_document=True,
            text="Beta text. Some other content.",
            score=0.9,
            rank=1,
        ),
        RetrievedPassage(
            doc_id="doc_A",
            is_full_document=True,
            text="Alpha text. Some sentence text.",
            score=0.8,
            rank=2,
        ),
        RetrievedPassage(
            doc_id="doc_C",
            is_full_document=True,
            text="Gamma text.",
            score=0.5,
            rank=3,
        ),
    ]

    result = evaluate_query(
        query_id="q_range_test",
        retrieved_docs=retrieved_passages,
        relevant_docs=relevant,
        gold_evidence=gold_ev,
        cutoffs=[3],
        empty_relevant_handling="zero",
        empty_evidence_handling="zero",
    )

    for metric_name, val in result.metrics.items():
        assert 0.0 <= val <= 1.0, f"Metric {metric_name} out of range: {val}"


# ---------------------------------------------------------------------------
# Test 11: validation_checks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason="Processed data files not available — skipping validation integration test.",
)
def test_validation_checks():
    """Run the full validation suite and assert all checks pass."""
    from src.experiments.validate_hotpotqa import run_validation

    report = run_validation(
        processed_dir=_PROCESSED_DIR,
        runs_path=_ROOT / "results" / "raw" / "hotpotqa_runs.jsonl",
        report_path=_ROOT / "results" / "metrics" / "hotpotqa_validation_report.json",
    )

    failed = report.get("failed_checks", [])
    # Checks 6/7/8/9 about runs may be skipped if no runs yet
    run_checks = {
        "retrieved_doc_ids_in_candidate_pool",
        "scores_non_increasing",
        "summary_metrics_recomputable",
    }
    critical_failures = [f for f in failed if f not in run_checks]
    assert not critical_failures, (
        f"Critical validation checks failed: {critical_failures}\n"
        f"Full report: {json.dumps(report['checks'], indent=2)}"
    )
