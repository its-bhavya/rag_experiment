# Research Paper & Reproducibility Notes

This directory stores research writeups, LaTeX templates, and experimental reports addressing the core research question:
> **Does conventional document-level retrieval evaluation adequately reflect a retrieval system's ability to retrieve all supporting evidence required to answer a question?**

---

## 1. Key Empirical Findings ($K \in \{3, 5, 7, 9\}$)

Experiments were conducted across two distinct benchmarks representing single-premise scientific claim verification (**SciFact**) and multi-hop reasoning over encyclopedic text (**HotpotQA**).

### SciFact (Open Corpus Retrieval: 5,183 Abstracts)
- **Document Recall vs. Evidence Completeness:** In scientific claim verification where a single document typically holds the needed rationale, Document Recall and Evidence Completeness track relatively closely, but strict sentence-level coverage is still required.
- **Dense vs. BM25:** Dense retrieval (`BAAI/bge-base-en-v1.5`) significantly outperforms BM25 at all cutoffs:
  - At $K=3$: Dense achieves **85.1%** Evidence Completeness vs. **78.7%** for BM25.
  - At $K=9$: Dense achieves **96.8%** Evidence Completeness vs. **88.8%** for BM25.

### HotpotQA (Multi-Hop Distractor Pool: 10 Paragraphs/Query)
- **The Core Discrepancy:** In multi-hop reasoning where multiple disjoint documents are required, **Document Recall severely overestimates whether the complete evidence chain has been recovered**:
  - At $K=3$: BM25 achieves a Document Recall of **67.8%**, but its Evidence Completeness is only **41.4%**—a **26.4 percentage-point deficit**. A retriever often finds one relevant hop while missing the second, leaving the downstream reader unable to answer.
  - At $K=5$: BM25 Document Recall is **80.4%**, yet Evidence Completeness is only **63.6%** (a 16.8-point gap).
  - Even with Dense retrieval at $K=3$, Document Recall is **85.4%** while Evidence Completeness is **72.0%** (a 13.4-point gap).
- **Takeaway for the Paper:** Conventional IR benchmarks report Document Recall and treat a query as successful if top-$K$ contains relevant documents. However, for multi-hop RAG, **Document Recall gives a false sense of retrieval sufficiency**. Evidence Completeness@K is a necessary and more faithful metric.

---

## 2. Benchmark Summary Tables

### Table 1: SciFact Benchmark ($N=300$, Evaluable Evidence Queries $=188$)

| Method | $K$ | Recall@$K$ | Precision@$K$ | MRR | Evidence Recall@$K$ | Evidence Completeness@$K$ | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25** | 3 | 0.6375 | 0.2256 | 0.5750 | 0.8032 | 0.7872 | 25.4 |
| **BM25** | 5 | 0.6969 | 0.1500 | 0.5888 | 0.8621 | 0.8404 | 25.4 |
| **BM25** | 7 | 0.7358 | 0.1148 | 0.5946 | 0.9014 | 0.8830 | 25.4 |
| **BM25** | 9 | 0.7408 | 0.0900 | 0.5950 | 0.9067 | 0.8883 | 25.4 |
| **Dense** | 3 | 0.7401 | 0.2700 | 0.6767 | 0.8833 | 0.8511 | 64.1 |
| **Dense** | 5 | 0.8036 | 0.1787 | 0.6898 | 0.9319 | 0.9149 | 64.1 |
| **Dense** | 7 | 0.8429 | 0.1352 | 0.6957 | 0.9504 | 0.9415 | 64.1 |
| **Dense** | 9 | **0.8630** | 0.1085 | **0.6981** | **0.9752** | **0.9681** | 64.1 |
| **Hybrid** | 3 | 0.7183 | 0.2589 | 0.6483 | 0.8817 | 0.8617 | 160.0 |
| **Hybrid** | 5 | 0.7770 | 0.1700 | 0.6610 | 0.9219 | 0.8989 | 160.0 |
| **Hybrid** | 7 | 0.7945 | 0.1248 | 0.6641 | 0.9386 | 0.9202 | 160.0 |
| **Hybrid** | 9 | 0.8200 | 0.1011 | 0.6670 | 0.9466 | 0.9255 | 160.0 |

---

### Table 2: HotpotQA Distractor Benchmark ($N=500$, Closed Pool: 10 Paras/Query)

| Method | $K$ | Recall@$K$ | Precision@$K$ | MRR | Evidence Recall@$K$ | Evidence Completeness@$K$ | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25** | 3 | 0.6780 | 0.4520 | 0.8670 | 0.6809 | 0.4140 | 0.66 |
| **BM25** | 5 | 0.8040 | 0.3216 | 0.8736 | 0.8078 | 0.6360 | 0.64 |
| **BM25** | 7 | 0.8660 | 0.2474 | 0.8762 | 0.8679 | 0.7440 | 0.64 |
| **BM25** | 9 | 0.9330 | 0.2073 | 0.8776 | 0.9350 | 0.8660 | 0.66 |
| **Dense** | 3 | 0.8540 | 0.5693 | 0.9353 | 0.8551 | 0.7200 | 55.9 |
| **Dense** | 5 | 0.9270 | 0.3708 | 0.9366 | 0.9290 | 0.8600 | 0.04 |
| **Dense** | 7 | 0.9630 | 0.2751 | 0.9373 | 0.9636 | 0.9280 | 0.03 |
| **Dense** | 9 | 0.9810 | 0.2180 | **0.9375** | 0.9812 | 0.9620 | 0.04 |
| **Hybrid** | 3 | 0.8050 | 0.5367 | 0.9287 | 0.8082 | 0.6340 | 0.82 |
| **Hybrid** | 5 | 0.9010 | 0.3604 | 0.9332 | 0.9025 | 0.8060 | 0.76 |
| **Hybrid** | 7 | 0.9470 | 0.2706 | 0.9335 | 0.9478 | 0.8960 | 0.74 |
| **Hybrid** | 9 | **0.9830** | 0.2184 | 0.9338 | **0.9825** | **0.9660** | 0.74 |

---

## 3. Scientific Integrity & Validation Checklist
- **No Test Leakage:** Queries were ranked without access to qrels or gold supporting-fact annotations.
- **Deterministic Evaluation:** Ranks, ties, and normalizations are fully deterministic.
- **Strict Evidence Coverage:** Passages are credited with evidence coverage only when the verified evidence sentence text actually appears within the candidate passage text (`is_evidence_covered_by_passage`).
- **Reproducibility Reports:**
  - SciFact: `results/metrics/validation_report.json` (10/10 passed, `safe_to_use_in_paper: true`).
  - HotpotQA: `results/metrics/hotpotqa_validation_report.json` (9/9 passed).
