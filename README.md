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
    processed/
      scifact/              # Clean JSONL artifacts with preserved metadata & evidence
  src/
    datasets/
      scifact.py            # SciFact loader, parser, and exporter
      inspect_dataset.py    # Diagnostic inspection script
    retrieval/              # BM25, Dense, and Hybrid (RRF) retrieval methods
    evaluation/
      metrics.py            # Document- and evidence-level retrieval metrics
    experiments/            # Experiment execution pipelines and harness
  results/
    raw/                    # Per-query retrieval runs and raw inspection dumps
    metrics/                # Computed metric summaries (CSV/JSON)
    tables/                 # Formatted comparative tables
  figures/                  # Visualizations and metric discrepancy plots
  tests/
    test_metrics.py         # Unit tests for evaluation metrics
    test_datasets.py        # Unit tests for dataset parsing and integrity
  paper/                    # Scientific writeup, LaTeX notes, and findings
  requirements.txt          # Python dependencies
  pyproject.toml            # Project and testing configuration
  README.md                 # Setup, download, and execution instructions
  .gitignore                # Ignored cache, model checkpoints, and data archives
  .env.example              # Sample environment configuration
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

Formal mathematical definitions and denominator specifications are detailed in [`docs/metrics_definition.md`](docs/metrics_definition.md):

1. **Document-Level Metrics:**
   - **Recall@K** ($K$ configurable, e.g. $\{5, 10\}$): Fraction of gold relevant documents retrieved in top-$K$. Denominator is $|D^*|$ (total relevant docs for the query).
   - **Precision@K** ($K$ configurable): Fraction of top-$K$ retrieved slots that are relevant. Denominator is strictly $K$. When fewer than $K$ documents are available, unretrieved slots are treated as non-relevant.
   - **MRR (Mean Reciprocal Rank)**: Reciprocal rank of the first relevant retrieved document ($1/\text{rank}_1$). Queries with no relevant result receive $0.0$.
2. **Evidence-Level Metrics:**
   - **Evidence Recall@K**: Fraction of annotated supporting evidence units covered by retrieved passages. Denominator is $|E^*|$ (total annotated evidence units for the query).
   - **Evidence Completeness@K**: Strict binary flag ($1.0$ or $0.0$) indicating whether **every** required evidence unit is covered within top-$K$. Aggregated as the fraction of evaluable questions with complete evidence.

**Passage & Context Coverage Semantics:**
- A document being retrieved does **not** automatically imply all of its supporting sentences were retrieved unless the retrieval context actually includes those sentences.
- Exact and normalized annotation matching (NFKD unicode, quote unifications, dash unifications, punctuation stripping) is used—never semantic similarity invented post-retrieval.
- When retrieval operates on passages or chunks, `map_passages_to_documents` projects them back to parent document IDs (preserving rank of first appearance) for document metrics, while preserving full document-to-passage relationships in `QueryEvaluation.doc_to_passages`.

---

## Phase 5: Reproducible Experiment Runner

Run the full comparative evaluation across sparse (BM25), dense (BAAI/bge-base-en-v1.5 + FAISS), and hybrid (RRF, $c=60$) retrieval over all official SciFact test queries:

```bash
# Full execution across BM25, Dense, and Hybrid for K=5 and K=10:
python -m src.experiments.run_experiments --dataset scifact

# Fast single-method execution:
python -m src.experiments.run_experiments --dataset scifact --methods bm25

# Rerun from scratch (overwrites previous runs):
python -m src.experiments.run_experiments --dataset scifact --overwrite
```

### Generated Artifacts
- `results/raw/scifact_runs.jsonl`: 1,800 detailed retrieval records preserving query IDs, retrieved document IDs, scores, latencies, evidence items recovered, and query-level metrics.
- `results/metrics/scifact_per_query.csv`: Per-query metric evaluation table for all (query, method, K) runs.
- `results/metrics/scifact_summary.csv`: Aggregated comparative summary table.
- `results/metrics/scifact_run_metadata.json`: Full environment provenance, dataset SHA-256 fingerprints, random seed, and library versions.

### SciFact Evaluation Results Summary

| Method | K | Query Count | Recall@K | Precision@K | MRR | Evidence Recall@K | Evidence Completeness@K | Mean Latency (ms) | Median Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| **BM25** | 5 | 300 | 0.6969 | 0.1500 | 0.5888 | 0.8621 | 0.8404 | 24.21 | 23.01 |
| **BM25** | 10 | 300 | 0.7598 | 0.0833 | 0.5970 | 0.9120 | 0.8936 | 24.21 | 23.01 |
| **Dense** | 5 | 300 | 0.8036 | 0.1787 | 0.6898 | 0.9319 | 0.9149 | 71.25 | 68.45 |
| **Dense** | 10 | 300 | 0.8703 | 0.0987 | 0.6987 | 0.9752 | 0.9681 | 71.25 | 68.45 |
| **Hybrid** | 5 | 300 | 0.7770 | 0.1700 | 0.6610 | 0.9219 | 0.8989 | 103.46 | 98.27 |
| **Hybrid** | 10 | 300 | 0.8283 | 0.0920 | 0.6676 | 0.9466 | 0.9255 | 103.46 | 98.27 |

---

## Running Unit Tests

Run the test suite with `pytest`:

```bash
python -m pytest tests -v
```

All 73 unit tests validate document-level metrics, evidence completeness edge cases, multi-hop evidence spread, ranking validation, passage-to-document mapping, context-sensitive evidence matching, BM25/Dense/Hybrid retrieval correctness, RRF math, experiment runner reproducibility, resumability, and dataset parsing integrity.

> **Note:** Dense retrieval tests (`TestDenseRetriever`) require FAISS to load successfully. They are automatically skipped on systems where the FAISS DLL is unavailable; all other tests (BM25, RRF, Hybrid, metrics, dataset) run unconditionally.
