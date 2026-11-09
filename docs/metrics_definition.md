# Metric Definitions and Evaluation Protocol

This document formally specifies all retrieval evaluation metrics used to evaluate sparse, dense, and hybrid retrieval systems for Evidence Completeness in Information Retrieval.

---

## 1. Notation and Terminology

- Let $Q$ denote the evaluation query set.
- For each query $q \in Q$:
  - $R_q = [p_1, p_2, \dots, p_N]$ denotes the ordered sequence of retrieved items (documents, passages, or sentence chunks) returned by a retrieval system for query $q$.
  - $R_q@K = [p_1, \dots, p_K]$ denotes the top-$K$ retrieved items ($K \le N$).
  - $\text{doc}(p)$ maps a retrieved passage $p$ to its parent document ID.
  - $\text{docs}(R_q@K)$ maps top-$K$ passages back to unique parent document IDs in first-appearance rank order.
  - $D^*_q$ denotes the set of gold relevant documents for query $q$ (as defined by dataset qrels).
  - $E^*_q = \{e_1, e_2, \dots, e_M\}$ denotes the set of discrete gold evidence annotations (e.g., supporting sentences or facts) required to substantiate or answer $q$.
  - $\text{doc}(e)$ maps an evidence item $e \in E^*_q$ to its parent document ID.
  - $\text{text}(e)$ denotes the canonical text string of evidence sentence $e$.

---

## 2. Conventional Document-Level Metrics

### 2.1 Recall@K (Recall@5, Recall@10)
**Definition:**  
The proportion of gold relevant documents retrieved within the top-$K$ ranks.

$$\text{Recall@K}(q) = \frac{|\text{docs}(R_q@K) \cap D^*_q|}{|D^*_q|}$$

- **Range:** $[0.0, 1.0]$
- **Denominator Choice:** $|D^*_q|$ (total count of unique gold relevant documents for query $q$).
- **Edge cases:**
  - If $|D^*_q| = 0$: in strict mode raises an explicit `ValueError`; when `empty_relevant_handling="zero"`, returns $0.0$.
  - If $|R_q| = 0$ (retriever returns nothing), $\text{Recall@K}(q) = 0.0$.
  - $K$ is fully configurable ($K \in \{1, 3, 5, 10, 20\}$).

### 2.2 Precision@K (Precision@5, Precision@10)
**Definition:**  
The proportion of top-$K$ retrieved slots that are gold relevant documents.

$$\text{Precision@K}(q) = \frac{|\text{docs}(R_q@K) \cap D^*_q|}{K}$$

- **Range:** $[0.0, 1.0]$
- **Denominator Choice:** Strictly $K$ (fixed rank cutoff, standard IR/TREC definition).
- **Behavior When Fewer Than K Documents Are Available:**
  When a retrieval system returns fewer than $K$ candidates (e.g. only 2 documents returned when $K=5$), the unretrieved positions ($K - |R_q|$) are treated as non-relevant documents. The denominator remains $K$.
  $$\text{Example: } 2 \text{ relevant documents retrieved out of } 2 \text{ returned for } K=5 \implies \text{Precision@5} = \frac{2}{5} = 0.40$$

### 2.3 Mean Reciprocal Rank (MRR)
**Definition:**  
The reciprocal of the rank of the first relevant document retrieved for query $q$.

$$\text{RR}(q) = \begin{cases} \frac{1}{\text{rank}_1(q)} & \text{if } \exists d_i \in \text{docs}(R_q) \text{ s.t. } d_i \in D^*_q \\ 0.0 & \text{otherwise} \end{cases}$$

where $\text{rank}_1(q) = \min \{i \in [1, |\text{docs}(R_q)|] \mid d_i \in D^*_q\}$.  
- **Queries with no relevant result receive zero.**
- The corpus-level metric is macro-averaged across queries:
$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}(q)$$

---

## 3. Evidence-Level Metrics

### 3.1 Evidence Coverage Semantics
> **Critical Requirement:** A document being retrieved must **not** automatically imply all of its supporting sentences were retrieved unless the retrieval context actually includes those sentences.

An evidence item $e \in E^*_q$ is covered by a retrieved candidate $p \in R_q@K$ if and only if:
1. $\text{doc}(p) == \text{doc}(e)$ (parent document ID matches); AND
2. EITHER:
   - $p$ represents a full document ($p.\text{is\_full\_document} = \text{True}$) with no passage truncation; OR
   - The retrieved context text $\text{text}(p)$ contains the evidence sentence $\text{text}(e)$ verified via exact substring matching or normalized text matching; OR
   - The passage identifier $p.\text{passage\_id}$ exactly matches $e.\text{evidence\_id}$.

If a retrieved passage belongs to document $\text{doc}(e)$ but its text does *not* contain the sentence $\text{text}(e)$, the evidence item is **not covered**.

### 3.2 Exact & Normalized Text Matching Protocol
Evidence matching strictly uses deterministic string normalization:
1. **Unicode NFKD normalization** (unifies decomposed characters and accents).
2. **Punctuation standardization** (converts em-dashes, en-dashes, and unicode hyphens to `-`; strips smart quotes `'` and `"`).
3. **Whitespace collapsing** (converts all whitespace sequences to a single space).
4. **Case normalization** (lowercasing).
5. **Exact token sequence fallback** (verifies sequential token alignment without semantic hallucination).

*No LLM generation or semantic embedding similarity is used during evidence verification.*

### 3.3 Evidence Recall@K (Evidence Recall@5, Evidence Recall@10)
**Definition:**  
The proportion of all annotated gold supporting evidence units covered by the top-$K$ retrieved passages.

$$E_{\text{retrieved}}@K(q) = \{e \in E^*_q \mid \exists p \in R_q@K \text{ that covers } e\}$$

$$\text{EvidenceRecall@K}(q) = \frac{|E_{\text{retrieved}}@K(q)|}{|E^*_q|}$$

- **Range:** $[0.0, 1.0]$
- **Denominator Choice:** $|E^*_q|$ (total count of annotated gold evidence units for query $q$).
- **Rationale:** Measures fractional recovery of the necessary supporting evidence pieces.

### 3.4 Evidence Completeness@K (Evidence Completeness@5, Evidence Completeness@10)
**Definition:**  
A strict all-or-nothing binary indicator for a single query indicating whether **every** required gold evidence piece is present in the top-$K$ retrieved passages.

$$\text{EvidenceCompleteness@K}(q) = \begin{cases} 1.0 & \text{if } E_{\text{retrieved}}@K(q) = E^*_q \ (\text{i.e. } \text{EvidenceRecall@K}(q) = 1.0) \\ 0.0 & \text{otherwise} \end{cases}$$

The dataset-level score is the macro-averaged fraction of evaluable questions with complete supporting evidence retrieved:

$$\text{EvidenceCompleteness@K} = \frac{1}{|Q|} \sum_{q \in Q} \text{EvidenceCompleteness@K}(q)$$

- **Research Relevance:** This metric directly tests the core research question:
  > *"Does conventional document-level retrieval evaluation adequately reflect a retrieval system's ability to retrieve all supporting evidence required to answer a question?"*
  A system may achieve high Recall@5 (e.g. 0.8) by finding single pieces of evidence across queries, yet fail on Evidence Completeness (e.g. 0.2) because multi-hop or multi-evidence questions frequently miss the second or third required premise.

---

## 4. Passage-to-Document Mapping & Relationship Preservation

When retrieval operates on passages or sentence chunks:
1. **Passage-to-Document Mapping:**
   `map_passages_to_documents(passages)` maps passages to parent document IDs, deduplicating subsequent occurrences of the same document while strictly preserving the rank of its first appearance.
2. **Preservation of Context:**
   `preserve_doc_to_passages(passages)` retains the full list of passages associated with each document ID in `QueryEvaluation.doc_to_passages`.
3. **Document Metrics Evaluation:**
   Document-level metrics ($\text{Recall@K}$, $\text{Precision@K}$, $\text{MRR}$) are evaluated on the projected parent document rankings.
4. **Evidence Metrics Evaluation:**
   Evidence metrics ($\text{EvidenceRecall@K}$, $\text{EvidenceCompleteness@K}$) are evaluated directly on the retrieved passage contexts.

---

## 5. Input Integrity & Error Handling Constraints

1. **No Silent Drops:** Malformed queries, invalid rank cutoffs ($K \le 0$), or empty document IDs raise explicit `ValueError` exceptions.
2. **Deterministic Tie-Breaking:** All rankings and deduplications are completely deterministic.
3. **No Duplicate Retrieved IDs in Document Rankings:** Raw document ID rankings with duplicates are detected and rejected unless explicit passage deduplication mapping is used.
4. **Machine-Readable Outputs:** Every per-query evaluation and macro-aggregate is exportable to JSON and CSV.
