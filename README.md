# Evidence Completeness in Information Retrieval

A reproducible research project evaluating whether conventional document-level retrieval evaluation adequately reflects a retrieval system's ability to retrieve all supporting evidence required to answer complex questions and verify scientific claims.

## Research Question
> **Does conventional document-level retrieval evaluation adequately reflect a retrieval system's ability to retrieve all supporting evidence required to answer a question?**

---

## Project Structure

```
rag-retrieval-study/
  data/
    raw/
      scifact/              # Raw BEIR SciFact archive, corpus, queries, qrels
      scifact_official/     # Official SciFact repository release with rationale annotations
      hotpotqa/             # Official HotpotQA distractor validation release
    processed/
      scifact/              # Clean SciFact JSONL artifacts
      scifact_evidence.jsonl # Sentence-level evidence ground truth for SciFact
      hotpotqa_questions.jsonl  # Sampled HotpotQA questions (N=500, seed 12345)
      hotpotqa_candidates.jsonl # Candidate distractor paragraphs (10 per question)
      hotpotqa_evidence.jsonl   # Exact sentence evidence mapping for HotpotQA
  src/
    datasets/
      scifact.py            # SciFact loader, parser, and exporter
      evidence_mapper.py    # SciFact rationale sentence alignment mapper
      hotpotqa.py           # HotpotQA distractor loader, sampler, and evidence mapper
      inspect_dataset.py    # Diagnostic inspection script
    retrieval/              # BM25, Dense (BGE-base-en-v1.5 + FAISS), and Hybrid (RRF)
    evaluation/
      metrics.py            # Document- and evidence-level retrieval metrics
    experiments/
      run_experiments.py    # SciFact reproducible retrieval and evaluation runner
      validate_experiments.py # 10-point integrity and sanity checker for SciFact
      run_hotpotqa.py       # HotpotQA distractor retrieval and evaluation runner
      validate_hotpotqa.py  # 9-point integrity and sanity checker for HotpotQA
  results/
    raw/                    # Per-query retrieval runs and raw inspection dumps
    metrics/                # Computed metric summaries (CSV/JSON) and audit reports
    tables/                 # Formatted comparative tables
  figures/                  # Visualizations and metric discrepancy plots
  tests/
    test_metrics.py         # Unit tests for evaluation metrics
    test_datasets.py        # Unit tests for dataset parsing and integrity
    test_retrieval.py       # Unit tests for BM25, Dense, and Hybrid retrievers
    test_experiment_validation.py # Unit tests for SciFact experiment validation
    test_hotpotqa.py        # Unit tests for HotpotQA pipeline, metrics, and validation
  paper/                    # Scientific writeup, LaTeX notes, and findings
  requirements.txt          # Python dependencies
  pyproject.toml            # Project and testing configuration
  README.md                 # Setup, download, and execution instructions
```

---

## Setup & Environment

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12.8)
- Virtual environment recommended:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset: SciFact

SciFact evaluates claim verification over scientific abstracts from the biomedical literature. Each claim is annotated with supporting/contradicting rationale sentences.

### 2. Download and Extraction
To download the official BEIR SciFact dataset release into `data/raw/scifact/` with automated checksum verification:

```bash
python -c "from src.datasets.scifact import download_scifact; download_scifact()"
```

*Source URL:* `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`  
*SHA256 Checksum:* `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`

### 3. Inspection & Processing
Run the dataset inspection diagnostic script to audit dataset counts, integrity (missing IDs, empty documents, duplicates), document length distributions, and supporting evidence annotations:

```bash
# Run diagnostic inspection on test split
python -m src.datasets.inspect_dataset --split test

# Run diagnostic inspection AND export processed artifacts to data/processed/scifact/
python -m src.datasets.inspect_dataset --split test --process
```

### 4. Evidence Ground Truth Mapping (Phase 2)
To generate reliable evidence-level ground truth mapping official supporting sentences to BEIR corpus documents using robust normalized text matching:

```bash
python -c "
from src.datasets.evidence_mapper import build_evidence_mappings, save_evidence_mappings
records, report = build_evidence_mappings(
    'data/raw/scifact_official/data/claims_dev.jsonl',
    'data/raw/scifact_official/data/corpus.jsonl',
    'data/raw/scifact/corpus.jsonl'
)
save_evidence_mappings(records, report)
"
```

Artifacts generated:
- `data/processed/scifact_evidence.jsonl`: Discrete evidence records preserving `query_id`, `evidence_doc_id`, `sentence_idx`, `evidence_sentence`, `evidence_annotation_id`, `evidence_match_status`, and `match_method`.
- `data/processed/evidence_mapping_report.json`: Diagnostic report covering mapping success rates and query completeness.

#### Mapping Diagnostic Summary:
- **Total Evidence Annotations:** 366 sentences across 188 claims
- **Successfully Mapped:** 366 (100% success rate: 336 exact literal matches, 30 normalized matches)
- **Unmatched Evidence:** 0
- **Queries with Complete Evidence:** 188 (100% of claims with evidence)
- **Queries with Partial/Zero Evidence:** 0

#### Diagnostic Summary:
- **Corpus Documents:** 5,183
- **Total Queries:** 1,109
- **Test Qrels Queries:** 300 (339 relevance pairs)
- **Empty Documents:** 0
- **Duplicate Document / Query IDs:** None
- **Missing IDs in Qrels:** 0
- **Document Length (Words):** Mean = 214.6 ± 87.3, Median = 204.0 (Min: 33, Max: 1,541)
- **Supporting Evidence:** 188 of 300 test queries have 366 sentence-level evidence annotations.

---

## Phase 3: Retrieval Systems

Three retrieval methods are implemented under `src/retrieval/`, all operating on identical SciFact documents and queries. Qrels are never used during retrieval.

### Retrieval Implementations

| Method | Implementation | Key Parameters |
|--------|---------------|----------------|
| **BM25** | `rank_bm25.BM25Okapi` | k1=1.5, b=0.75; unicode-normalized tokenizer |
| **Dense** | `BAAI/bge-base-en-v1.5` + FAISS `IndexFlatIP` | 768-dim, L2-normalized, cosine sim = inner product |
| **Hybrid** | Reciprocal Rank Fusion (RRF) | c=60, candidate_pool_size=100 |

**RRF formula:** `RRF(d) = Σ 1/(c + rank_s(d))` summed over all retrieval systems s.  
**Missing documents:** A document absent from one system's candidate list contributes 0 to that system's RRF term. Deduplication is applied before final ranking.

### 5. Build Indexes

Build and cache both indexes before running queries. This downloads `BAAI/bge-base-en-v1.5` on first run (~435 MB):

```bash
# Build both BM25 and Dense FAISS indexes (may take several minutes for dense encoding)
python src/experiments/build_indexes.py

# BM25 only (fast, ~1 second):
python src/experiments/build_indexes.py --skip-dense

# Custom model or device:
python src/experiments/build_indexes.py --model-id BAAI/bge-base-en-v1.5 --device cpu
```

**Output artifacts:**
- `results/raw/bm25_index.pkl` — Serialized BM25Okapi index
- `results/raw/dense_index/` — FAISS index, embeddings (`.npy`), doc IDs, metadata
- `results/raw/index_build_log.json` — Timing and configuration record

### 6. Single-Query CLI

Retrieve top-K documents from one method for a given query:

```bash
# BM25 retrieval
python scripts/query.py --query "myeloid-derived suppressor cells" --method bm25 --k 5

# Dense retrieval
python scripts/query.py --query "myeloid-derived suppressor cells" --method dense --k 5

# Hybrid (RRF) retrieval
python scripts/query.py --query "myeloid-derived suppressor cells" --method hybrid --k 5
```

**Example output:**
```
======================================================================
Method : BM25
Query  : myeloid-derived suppressor cells
======================================================================
Rank   Score     Doc ID        Title
----------------------------------------------------------------------
1      5.3421    4983          Induction of myelodysplasia by myelo...
2      3.1200    2817          Myeloid cell suppressor mechanisms i...
...
```

### 7. Side-by-Side Comparison CLI

Compare all three methods for the same query in one command:

```bash
python scripts/compare.py --query "myeloid-derived suppressor cells" --k 5
```

---

## Phase 4: Evaluation Metrics

Formal mathematical definitions and denominator specifications are detailed in [`docs/metrics_definition.md`](docs/metrics_definition.md). Primary experimental cutoffs evaluated across both benchmarks are $K \in \{3, 5, 7, 9\}$:

1. **Document-Level Metrics:**
   - **Recall@K** ($K \in \{3, 5, 7, 9\}$): Fraction of gold relevant documents retrieved in top-$K$. Denominator is $|D^*|$ (total relevant docs for the query).
   - **Precision@K** ($K \in \{3, 5, 7, 9\}$): Fraction of top-$K$ retrieved slots that are relevant. Denominator is strictly $K$. When fewer than $K$ documents are available, unretrieved slots are treated as non-relevant.
   - **MRR (Mean Reciprocal Rank)**: Reciprocal rank of the first relevant retrieved document ($1/\text{rank}_1$). Queries with no relevant result receive $0.0$.
2. **Evidence-Level Metrics:**
   - **Evidence Recall@K**: Fraction of annotated supporting evidence units covered by retrieved passages. Denominator is $|E^*|$ (total annotated evidence units for the query).
   - **Evidence Completeness@K**: Strict binary flag ($1.0$ or $0.0$) indicating whether **every** required evidence unit is covered within top-$K$. Aggregated as the macro-averaged fraction of evaluable questions with complete evidence.

**Passage & Context Coverage Semantics:**
- A document being retrieved does **not** automatically imply all of its supporting sentences were retrieved unless the retrieval context actually includes those sentences.
- Exact and normalized annotation matching (NFKD unicode, quote unifications, dash unifications, punctuation stripping) is used—never semantic similarity invented post-retrieval.
- When retrieval operates on passages or chunks, `map_passages_to_documents` projects them back to parent document IDs (preserving rank of first appearance) for document metrics, while preserving full document-to-passage relationships in `QueryEvaluation.doc_to_passages`.

---

## Phase 5: Reproducible SciFact Experiment Runner

Run the full comparative evaluation across sparse (BM25), dense (BAAI/bge-base-en-v1.5 + FAISS), and hybrid (RRF, $c=60$) retrieval over all official SciFact test queries for cutoffs $K \in \{3, 5, 7, 9\}$:

```bash
# Full execution across BM25, Dense, and Hybrid for K=3, 5, 7, 9:
python -m src.experiments.run_experiments --dataset scifact --k-values 3 5 7 9

# Fast single-method execution:
python -m src.experiments.run_experiments --dataset scifact --methods bm25 --k-values 3 5 7 9

# Rerun from scratch (overwrites previous runs):
python -m src.experiments.run_experiments --dataset scifact --k-values 3 5 7 9 --overwrite
```

### Generated Artifacts
- `results/raw/scifact_runs.jsonl`: 3,600 detailed retrieval records preserving query IDs, retrieved document IDs, scores, latencies, evidence items recovered, and query-level metrics (300 queries × 3 methods × 4 cutoffs).
- `results/metrics/scifact_per_query.csv`: Per-query metric evaluation table for all 3,600 runs.
- `results/metrics/scifact_summary.csv`: Aggregated comparative summary table.
- `results/metrics/scifact_run_metadata.json`: Full environment provenance, dataset SHA-256 fingerprints, random seed, and library versions.

### SciFact Evaluation Results Summary ($N=300$, Evaluable Evidence Queries $=188$)

| Method | K | Query Count | Recall@K | Precision@K | MRR | Evidence Recall@K | Evidence Completeness@K | Mean Latency (ms) | Median Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| **BM25** | 3 | 300 | 0.6375 | 0.2256 | 0.5750 | 0.8032 | 0.7872 | 25.45 | 23.90 |
| **BM25** | 5 | 300 | 0.6969 | 0.1500 | 0.5888 | 0.8621 | 0.8404 | 25.45 | 23.90 |
| **BM25** | 7 | 300 | 0.7358 | 0.1148 | 0.5946 | 0.9014 | 0.8830 | 25.45 | 23.90 |
| **BM25** | 9 | 300 | 0.7408 | 0.0900 | 0.5950 | 0.9067 | 0.8883 | 25.45 | 23.90 |
| **Dense** | 3 | 300 | 0.7401 | 0.2700 | 0.6767 | 0.8833 | 0.8511 | 64.08 | 37.14 |
| **Dense** | 5 | 300 | 0.8036 | 0.1787 | 0.6898 | 0.9319 | 0.9149 | 64.08 | 37.14 |
| **Dense** | 7 | 300 | 0.8429 | 0.1352 | 0.6957 | 0.9504 | 0.9415 | 64.08 | 37.14 |
| **Dense** | 9 | 300 | 0.8630 | 0.1085 | 0.6981 | 0.9752 | 0.9681 | 64.08 | 37.14 |
| **Hybrid** | 3 | 300 | 0.7183 | 0.2589 | 0.6483 | 0.8817 | 0.8617 | 159.99 | 103.92 |
| **Hybrid** | 5 | 300 | 0.7770 | 0.1700 | 0.6610 | 0.9219 | 0.8989 | 159.99 | 103.92 |
| **Hybrid** | 7 | 300 | 0.7945 | 0.1248 | 0.6641 | 0.9386 | 0.9202 | 159.99 | 103.92 |
| **Hybrid** | 9 | 300 | 0.8200 | 0.1011 | 0.6670 | 0.9466 | 0.9255 | 159.99 | 103.92 |

---

## Phase 6: Experiment Validation & Quality Checks (SciFact)

To guarantee scientific rigor prior to publication, the automated validation suite in `src/experiments/validate_experiments.py` verifies 10 sanity and integrity checks:

```bash
python -m src.experiments.validate_experiments
```

### Verified Checks (10/10 Passed)
1. **Query Count Check:** Exactly 300 unique test queries evaluated.
2. **Corpus Size Check:** Exactly 5,183 unique documents in BEIR corpus.
3. **Evidence Count Check:** Exactly 366 sentence-level evidence annotations.
4. **Evidence Mapping Sanity:** 0 unmapped evidence records (100% matched).
5. **Run Count Check:** Exactly 1,200 runs per method (3,600 total across BM25, Dense, Hybrid).
6. **No Duplicate Ranks:** Every retrieved ranking has distinct ranks $[1, \dots, K]$.
7. **Score Monotonicity:** Retrieval scores are non-increasing with rank.
8. **Positive Latencies:** All recorded per-query latencies are strictly positive.
9. **Evidence Metric Invariant:** $\text{EvidenceCompleteness@K} \le \text{EvidenceRecall@K}$ holds for all runs.
10. **Summary Recomputability:** 100% of summary metrics independently recompute from raw records within $10^{-4}$ tolerance.

Artifact generated: `results/metrics/validation_report.json` (`safe_to_use_in_paper: true`).

---

## Phase 7: HotpotQA Distractor Benchmark Extension

To evaluate evidence completeness in **multi-hop reasoning**, we evaluate sparse (BM25), dense (BAAI/bge-base-en-v1.5), and hybrid (RRF, $c=60$) retrieval on the official HotpotQA distractor validation split.

> [!WARNING]
> **Experimental Protocol Disclaimer:** This experiment operates on the **closed distractor setting** supplied with the official HotpotQA benchmark. For each question, retrieval ranks the exact **10 candidate paragraphs** (2 gold context paragraphs + 8 adversarial distractors) provided in the dataset. This is **not** open-domain full-Wikipedia retrieval.

### Protocol Details
- **Question Sampling:** 500 questions sampled from the official validation split with documented random seed `12345`.
- **Candidates:** 4,954 unique candidate paragraphs (10 per question, shared candidates across queries deduplicated in global store).
- **Supporting Facts:** 1,208 gold supporting facts mapped to pairs of `(title, sentence_idx)` with exact sentence text preserved.
- **Fair Comparison:** All three methods (BM25, Dense, Hybrid) receive identical candidate pools for every query. Qrels and answers are never accessed during ranking.

### Execution

```bash
# Run full HotpotQA evaluation for K=3, 5, 7, 9 (uses pre-cached candidate embeddings):
python -m src.experiments.run_hotpotqa --sample-size 500 --seed 12345 --k-values 3 5 7 9

# Run validation checks:
python -m src.experiments.validate_hotpotqa
```

### Generated Artifacts
- `results/raw/hotpotqa_runs.jsonl`: 6,000 detailed retrieval records (500 queries × 3 methods × 4 cutoffs).
- `results/metrics/hotpotqa_per_query.csv`: Per-query metrics for all 6,000 runs.
- `results/metrics/hotpotqa_summary.csv`: Aggregated comparative summary table.
- `results/metrics/hotpotqa_validation_report.json`: 9-point validation report (9/9 passed).

### HotpotQA Evaluation Results Summary ($N=500$, Closed Pool: 10 Paras/Query)

| Method | K | Query Count | Recall@K | Precision@K | MRR | Evidence Recall@K | Evidence Completeness@K | Mean Latency (ms) | Median Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| **BM25** | 3 | 500 | 0.6780 | 0.4520 | 0.8670 | 0.6809 | 0.4140 | 0.66 | 0.63 |
| **BM25** | 5 | 500 | 0.8040 | 0.3216 | 0.8736 | 0.8078 | 0.6360 | 0.64 | 0.60 |
| **BM25** | 7 | 500 | 0.8660 | 0.2474 | 0.8762 | 0.8679 | 0.7440 | 0.64 | 0.59 |
| **BM25** | 9 | 500 | 0.9330 | 0.2073 | 0.8776 | 0.9350 | 0.8660 | 0.66 | 0.60 |
| **Dense** | 3 | 500 | 0.8540 | 0.5693 | 0.9353 | 0.8551 | 0.7200 | 55.92 | 54.35 |
| **Dense** | 5 | 500 | 0.9270 | 0.3708 | 0.9366 | 0.9290 | 0.8600 | 0.04 | 0.03 |
| **Dense** | 7 | 500 | 0.9630 | 0.2751 | 0.9373 | 0.9636 | 0.9280 | 0.03 | 0.03 |
| **Dense** | 9 | 500 | 0.9810 | 0.2180 | 0.9375 | 0.9812 | 0.9620 | 0.04 | 0.04 |
| **Hybrid** | 3 | 500 | 0.8050 | 0.5367 | 0.9287 | 0.8082 | 0.6340 | 0.82 | 0.77 |
| **Hybrid** | 5 | 500 | 0.9010 | 0.3604 | 0.9332 | 0.9025 | 0.8060 | 0.76 | 0.70 |
| **Hybrid** | 7 | 500 | 0.9470 | 0.2706 | 0.9335 | 0.9478 | 0.8960 | 0.74 | 0.69 |
| **Hybrid** | 9 | 500 | 0.9830 | 0.2184 | 0.9338 | 0.9825 | 0.9660 | 0.74 | 0.69 |

---

## Key Research Findings: Document Recall vs. Evidence Completeness

Our empirical findings across both benchmarks directly address the core research question:

1. **Single-Premise vs. Multi-Hop Retrieval:**
   - On **SciFact** (single-premise scientific claims), Document Recall and Evidence Completeness track relatively closely because relevant rationale sentences reside within a single abstract.
   - On **HotpotQA** (multi-hop questions requiring disjoint Wikipedia articles), conventional Document Recall **drastically overestimates retrieval success**.
2. **The Completeness Deficit in Multi-Hop Retrieval:**
   - At $K=3$, BM25 achieves a Document Recall of **67.8%**, but its Evidence Completeness is only **41.4%**—a **26.4 percentage-point deficit**.
   - Retrievers frequently locate one relevant hop while missing the second, leaving the question unanswerable by a downstream reader even though conventional IR metrics report strong recall.
   - At $K=5$, the BM25 gap remains substantial: **80.4% Document Recall** vs. **63.6% Evidence Completeness** (16.8-point gap).
   - Even with dense retrieval at $K=3$, Document Recall is **85.4%** while Evidence Completeness is **72.0%** (13.4-point gap).
3. **Implication for RAG Evaluation:**
   - Document-level retrieval benchmarks provide a misleading picture of RAG readiness for multi-hop tasks. Evaluating **Evidence Completeness@K** is essential to avoid overestimating pipeline capabilities.

---

## Running Unit Tests

Run the full test suite with `pytest`:

```bash
python -m pytest tests -v
```

All **85 unit tests** validate:
- Document-level metrics (Recall, Precision, MRR) and boundary conditions.
- Evidence-level metrics (Evidence Recall, Evidence Completeness, multi-hop spread, all-or-nothing logic).
- Strict context-sensitive evidence matching (exact substring and NFKD normalized matching).
- Passage-to-document parent ID projection and rank deduplication.
- Sparse BM25, Dense (`bge-base-en-v1.5`), and Hybrid RRF retrieval correctness.
- SciFact experiment runner, checkpointing, and 10-point validation suite.
- HotpotQA distractor dataset loader, sampling determinism, evidence mapper, and 9-point validation suite.

> **Note:** Dense retrieval tests requiring FAISS (`TestDenseRetriever`) require the FAISS library to be available. On environments without native FAISS binaries, dense tests are gracefully skipped while all other 80+ tests run and pass unconditionally.
