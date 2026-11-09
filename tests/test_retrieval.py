"""Unit tests for retrieval systems: BM25, Dense, and Hybrid (RRF).

Dense tests require FAISS to be loadable. They are automatically skipped when
the FAISS DLL is unavailable (see conftest.py skip_if_no_faiss marker).
"""

from pathlib import Path
import numpy as np
import pytest

from src.retrieval.base import BaseRetriever, ScoredDoc
from src.retrieval.bm25 import BM25Retriever, tokenize_bm25
from src.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion

# BGE_QUERY_PREFIX imported lazily to avoid DLL failure at collection time
try:
    from src.retrieval.dense import DenseRetriever, BGE_QUERY_PREFIX  # noqa: F401
    _DENSE_AVAILABLE = True
except (ImportError, OSError):
    _DENSE_AVAILABLE = False
    DenseRetriever = None  # type: ignore[assignment,misc]
    BGE_QUERY_PREFIX = ""

from conftest import skip_if_no_faiss


class TestScoredDoc:
    def test_valid_scored_doc(self):
        doc = ScoredDoc(doc_id="doc1", score=0.85, rank=1)
        assert doc.doc_id == "doc1"
        assert doc.score == 0.85
        assert doc.rank == 1

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError, match="1-indexed"):
            ScoredDoc(doc_id="doc1", score=0.5, rank=0)
        with pytest.raises(ValueError, match="1-indexed"):
            ScoredDoc(doc_id="doc1", score=0.5, rank=-1)


class TestBM25Retriever:
    def test_tokenize_bm25(self):
        tokens = tokenize_bm25("COVID-19 vaccination reduces mortality (p < 0.001)!")
        assert "covid-19" in tokens or ("covid" in tokens and "19" in tokens)
        assert "vaccination" in tokens
        assert "mortality" in tokens

    def test_tokenize_empty(self):
        assert tokenize_bm25("") == []
        assert tokenize_bm25("   ") == []
        assert tokenize_bm25("---") == []

    def test_bm25_ranking_and_score_sorting(self):
        doc_ids = ["doc_irrelevant", "doc_relevant", "doc_partial"]
        doc_texts = [
            "Photosynthesis in plants involves chlorophyll and sunlight.",
            "Induction of myelodysplasia by myeloid-derived suppressor cells.",
            "Myelodysplasia syndromes and stem cell disorders.",
        ]
        retriever = BM25Retriever(doc_ids=doc_ids, doc_texts=doc_texts)

        results = retriever.retrieve("myelodysplasia suppressor cells", k=3)
        assert len(results) == 3
        # Ensure scores are strictly sorted descending
        assert results[0].score >= results[1].score >= results[2].score
        # Ensure ranks are 1, 2, 3
        assert [r.rank for r in results] == [1, 2, 3]
        # Most relevant document should be rank 1
        assert results[0].doc_id == "doc_relevant"

    def test_bm25_empty_document_safe(self):
        # Includes an empty document alongside several content documents.
        # Uses enough documents so BM25 IDF is meaningful (>2 docs needed for positive scores).
        doc_ids = ["doc_empty", "doc_oncology", "doc_cardio", "doc_neuro"]
        doc_texts = [
            "",
            "Valid biomedical research abstract on oncology and tumor suppression.",
            "Cardiovascular disease and hypertension treatment guidelines.",
            "Neuroplasticity in adult human brain after injury.",
        ]
        retriever = BM25Retriever(doc_ids=doc_ids, doc_texts=doc_texts)

        results = retriever.retrieve("oncology tumor biomedical", k=4)
        assert len(results) == 4
        # doc_oncology must rank first (most term overlap with query)
        assert results[0].doc_id == "doc_oncology"
        assert results[0].score > 0.0
        # Empty document must have score 0.0 (no terms to match)
        empty_result = next(r for r in results if r.doc_id == "doc_empty")
        assert empty_result.score == 0.0

    def test_bm25_empty_query(self):
        doc_ids = ["doc1", "doc2"]
        doc_texts = ["Sample text 1", "Sample text 2"]
        retriever = BM25Retriever(doc_ids=doc_ids, doc_texts=doc_texts)

        results = retriever.retrieve("   ", k=2)
        assert len(results) == 2
        assert results[0].score == 0.0
        assert results[1].score == 0.0

    def test_bm25_save_and_load(self, tmp_path: Path):
        save_file = tmp_path / "bm25_test.pkl"
        doc_ids = ["docA", "docB"]
        doc_texts = ["Gene expression analysis", "Protein folding dynamics"]
        retriever = BM25Retriever(doc_ids=doc_ids, doc_texts=doc_texts)
        retriever.save(save_file)

        loaded = BM25Retriever.load(save_file)
        res_orig = retriever.retrieve("gene expression", k=2)
        res_loaded = loaded.retrieve("gene expression", k=2)

        assert [r.doc_id for r in res_orig] == [r.doc_id for r in res_loaded]
        assert np.isclose(res_orig[0].score, res_loaded[0].score)


@skip_if_no_faiss
class TestDenseRetriever:
    def test_dense_retrieval_and_cosine_similarity(self):
        doc_ids = ["doc_neuro", "doc_cardio"]
        doc_texts = [
            "Neural microstructural development in infant brain white matter.",
            "Cardiovascular risk factors in chronic coronary disease.",
        ]
        retriever = DenseRetriever(device="cpu")
        retriever.build_index(doc_ids=doc_ids, doc_texts=doc_texts)

        results = retriever.retrieve("brain neural development", k=2)
        assert len(results) == 2
        assert results[0].doc_id == "doc_neuro"
        # Since vectors are unit normalized, inner product is in [-1.0, 1.0]
        assert -1.0 <= results[0].score <= 1.0
        assert results[0].score > results[1].score
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_dense_save_and_load(self, tmp_path: Path):
        save_dir = tmp_path / "dense_index"
        doc_ids = ["doc1", "doc2"]
        doc_texts = ["Cellular autophagy in yeast.", "Bacterial resistance to penicillin."]

        retriever = DenseRetriever(device="cpu")
        retriever.build_index(doc_ids=doc_ids, doc_texts=doc_texts, save_dir=save_dir)

        loaded = DenseRetriever.load(save_dir, device="cpu")
        res_orig = retriever.retrieve("autophagy", k=2)
        res_loaded = loaded.retrieve("autophagy", k=2)

        assert [r.doc_id for r in res_orig] == [r.doc_id for r in res_loaded]
        assert np.isclose(res_orig[0].score, res_loaded[0].score, atol=1e-5)


class TestReciprocalRankFusion:
    def test_rrf_scoring_math(self):
        # System 1: docA (rank 1), docB (rank 2)
        # System 2: docB (rank 1), docC (rank 2)
        # c = 60
        # docA score = 1/(60+1) + 0 = 1/61
        # docB score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        # docC score = 0 + 1/(60+2) = 1/62
        sys1 = [ScoredDoc("docA", 10.0, 1), ScoredDoc("docB", 5.0, 2)]
        sys2 = [ScoredDoc("docB", 0.9, 1), ScoredDoc("docC", 0.7, 2)]

        fused = reciprocal_rank_fusion([sys1, sys2], c=60, k=3)

        assert len(fused) == 3
        # docB appears in both systems -> highest score
        assert fused[0].doc_id == "docB"
        assert fused[0].rank == 1
        expected_docB_score = (1.0 / 62.0) + (1.0 / 61.0)
        assert np.isclose(fused[0].score, expected_docB_score)

        # docA appears at rank 1 in sys1 -> 1/61 > docC (rank 2 in sys2 -> 1/62)
        assert fused[1].doc_id == "docA"
        assert fused[1].rank == 2
        assert np.isclose(fused[1].score, 1.0 / 61.0)

        assert fused[2].doc_id == "docC"
        assert fused[2].rank == 3
        assert np.isclose(fused[2].score, 1.0 / 62.0)

    def test_rrf_missing_candidate_handling(self):
        # docA only present in sys1; docZ only present in sys2
        sys1 = [ScoredDoc("docA", 1.0, 1)]
        sys2 = [ScoredDoc("docZ", 1.0, 1)]

        fused = reciprocal_rank_fusion([sys1, sys2], c=60, k=2)
        assert len(fused) == 2
        # Both have score 1/61
        assert np.isclose(fused[0].score, 1.0 / 61.0)
        assert np.isclose(fused[1].score, 1.0 / 61.0)
        # Unique doc IDs preserved
        assert {fused[0].doc_id, fused[1].doc_id} == {"docA", "docZ"}

    def test_rrf_deduplication(self):
        # Multiple occurrences of docA across systems
        sys1 = [ScoredDoc("docA", 1.0, 1)]
        sys2 = [ScoredDoc("docA", 1.0, 1)]
        sys3 = [ScoredDoc("docA", 1.0, 1)]

        fused = reciprocal_rank_fusion([sys1, sys2, sys3], c=60, k=5)
        # Should contain docA exactly once
        assert len(fused) == 1
        assert fused[0].doc_id == "docA"
        assert np.isclose(fused[0].score, 3.0 / 61.0)

    def test_rrf_top_k_cutoff(self):
        sys1 = [ScoredDoc(f"doc_{i}", float(10 - i), i) for i in range(1, 10)]
        fused = reciprocal_rank_fusion([sys1], c=60, k=5)
        assert len(fused) == 5
        assert [d.rank for d in fused] == [1, 2, 3, 4, 5]

    def test_hybrid_retriever_pipeline(self):
        class MockRetriever(BaseRetriever):
            def __init__(self, doc_ids: list[str]):
                self.doc_ids = doc_ids

            def retrieve(self, query: str, k: int = 10) -> list[ScoredDoc]:
                return [ScoredDoc(doc_id, 1.0 / rank, rank) for rank, doc_id in enumerate(self.doc_ids[:k], 1)]

        ret_sparse = MockRetriever(["doc1", "doc2", "doc3"])
        ret_dense = MockRetriever(["doc2", "doc4", "doc1"])

        hybrid = HybridRetriever(ret_sparse, ret_dense, c=60, candidate_pool_size=10)
        res = hybrid.retrieve("query", k=3)

        assert len(res) == 3
        # doc2 is rank 2 in sparse, rank 1 in dense -> score: 1/62 + 1/61
        # doc1 is rank 1 in sparse, rank 3 in dense -> score: 1/61 + 1/63
        assert res[0].doc_id == "doc2"
        assert res[0].rank == 1
        assert res[1].doc_id == "doc1"
        assert res[1].rank == 2
