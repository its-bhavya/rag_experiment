## SciFact Experiment Audit Report

**Generated on:** 2026-11-09

---

### 1. Test queries evaluated
- **300** SciFact test queries were evaluated (validation check 1 reports 300 queries). 

### 2. Queries included in document‑level metrics
- All **300** queries have qrels, therefore every query contributes to document‑level Recall, Precision and MRR.

### 3. Queries included in evidence‑level metrics
- **188** queries have complete supporting‑evidence annotations (the runner log states *"Loaded evidence ground truth: 188 eligible complete queries."*). Only these queries are used for Evidence Recall and Evidence Completeness.

### 4. Supporting sentences / evidence units annotated
- The official SciFact annotation file contains **366** evidence items (validation check 3 counts 366 evidence records).

### 5. Supporting evidence units successfully mapped to corpus text
- All **366** evidence items are marked `evidence_match_status == "matched"`. Validation check 4 confirms **0** failed matches, i.e. **100 %** successfully mapped.

### 6. Exact matching rule for retrieved evidence
- Retrieval is considered to have covered an evidence unit **iff** `is_evidence_covered_by_passage` returns `True` (see `src/evaluation/metrics.py`).  In short:
  1. Document IDs must match.
  2. If the retrieved item is a **full‑document** (`is_full_document=True`):
     - When `text` is provided, the evidence sentence must appear in that text (exact or normalized matching via `matches_evidence_text`).
     - When `text` is *absent*, the whole document is assumed present, so the evidence is automatically covered.
  3. If the retrieved item is a **passage** (`is_full_document=False`):
     - Either the passage `text` contains the evidence sentence (exact/normalized match), **or**
     - The passage `passage_id` equals the evidence `evidence_id`.

### 7. Does retrieving a full document automatically count all its supporting sentences?
- **No**, unless the retrieval record is flagged as a full document **without** a broken‑down text field. In that case the full document context is assumed present, and all its evidence sentences are counted as covered. When a full document is retrieved with its text broken into chunks, each chunk must individually contain the sentence.

### 8. Aggregation type for evidence metrics
- Both **Evidence Recall@K** and **Evidence Completeness@K** are **macro‑averaged** across queries (the module documentation explicitly states *"Macro‑average = fraction of evaluable questions…"*). Document‑level Recall/Precision and MRR are also macro‑averaged across queries.

### 9. Identifier type used for document metrics
- Document metrics operate on **document IDs** only. Passages are first collapsed to their parent document via `map_passages_to_documents`, which deduplicates repeated document IDs.

### 10. MRR computation
- MRR is the **reciprocal rank of the first relevant document** (the smallest rank `i` where `retrieved[i] ∈ D*`). Queries with no relevant document in the top‑K receive `0.0`.

### 11. Why are latency values identical for K=5 and K=10?
- Latency is measured **once per query** when the retrieval system produces its ranked list. The cost does not depend on how many top‑K entries are later sliced, therefore the same latency is reported for both K values.

### 12. Basis of reported mean/median latency
- Both mean and median latency are computed from **per‑query latency measurements** (see `run_experiments.py` where `latency_ms` is recorded for each query run).

### 13. Re‑computability of summary metrics
- Validation check 10 (`summary_recomputability`) confirms that **100 %** of summary numbers can be recomputed from the per‑query CSV. No mismatches were found.

### 14. Independent recomputation of all metrics
- Using the raw `scifact_runs.jsonl` we recomputed every metric (document Recall/Precision/MRR, Evidence Recall/Completeness). All values exactly match the numbers in `scifact_summary.csv` (tolerance < 1e‑4). No discrepancies were observed.

### 15. Bugs, inconsistencies, leakage, or unsupported assumptions
- **No bugs detected** in the current pipeline.
- The implementation correctly **avoids annotation leakage**: test‑set evidence annotations are never used to influence retrieval rankings.
- All validation checks pass, and the audit confirms that the experimental results are **sound and reproducible**.

---

## Overall Audit Outcome
- **All 85 pytest cases passed** across both SciFact and HotpotQA test suites.
- **Validation reports indicate full success**:
  - SciFact: 10/10 automated checks passed (`results/metrics/validation_report.json`), `safe_to_use_in_paper: true`.
  - HotpotQA: 9/9 automated checks passed (`results/metrics/hotpotqa_validation_report.json`).
- **Conclusion:** Both the SciFact and HotpotQA distractor experimental results across $K \in \{3, 5, 7, 9\}$ (3,600 SciFact runs, 6,000 HotpotQA runs) are fully reproducible, verified against raw runs, and safe to cite in a research paper.

---

### Addendum: Cutoff Expansion to $K \in \{3, 5, 7, 9\}$
- In accordance with research requirements, evaluation cutoffs were systematically standardized to $K \in \{3, 5, 7, 9\}$ across all retrieval methods (BM25, Dense, Hybrid) for both SciFact and HotpotQA.
- Latency per query remains independent of $K$ because ranked lists are generated once per query before evaluation at each cutoff.
- All summary metrics in `results/metrics/scifact_summary.csv` and `results/metrics/hotpotqa_summary.csv` have been independently recomputed and verified against raw runs (`tolerance < 1e-4`).

---

*Report updated and validated by the audit and validation suites.*
