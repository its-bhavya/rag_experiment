"""Evaluation metrics for document retrieval and evidence completeness.

This module implements:
- Conventional document-level metrics: Recall@K, Precision@K, MRR (Mean Reciprocal Rank).
- Evidence-level metrics: Evidence Recall@K, Evidence Completeness@K.
- Passage-to-document mappings and document-to-passage relationship preservation.
- Exact and carefully normalized text matching for atomic evidence sentences.
- Per-query evaluation and dataset-level macro-average aggregation.

Metric Definitions and Denominator Choices:
------------------------------------------
1. Recall@K:
   Formula: |R@K ∩ D*| / |D*|
   Denominator: |D*| (total count of unique gold relevant documents for the query).
   Edge case: When |D*| == 0, behavior is governed by `empty_relevant_handling`:
     - "raise": raises ValueError (default in strict mode).
     - "zero": returns 0.0 (query has no relevant documents to recall).
   When retrieved list is empty (R = []), returns 0.0.

2. Precision@K:
   Formula: |R@K ∩ D*| / K
   Denominator: K (strictly fixed rank cutoff, standard IR/TREC definition).
   Fewer than K documents available: If the retrieval system returns fewer than K
   documents (e.g. 2 documents retrieved when K=5), unretrieved slots (K - |R|) are
   treated as non-relevant documents. The denominator remains K (e.g. 2 relevant
   docs out of 2 retrieved for K=5 yields 2/5 = 0.4).

3. Mean Reciprocal Rank (MRR):
   Formula: 1 / rank_1 where rank_1 = min { i ∈ [1, |R|] : d_i ∈ D* }.
   Queries with no relevant result in R (or within cutoff) receive 0.0.
   The dataset metric is the macro-average across all evaluated queries.

4. Evidence Recall@K:
   Formula: |E_retrieved@K| / |E*|
   Denominator: |E*| (total count of annotated gold evidence units for the query).
   An evidence unit e ∈ E* is covered by retrieved passages if there exists at least
   one retrieved passage p ∈ R@K such that:
     (a) p.doc_id == e.doc_id, AND
     (b) The retrieval context actually includes the evidence sentence:
         - If passage text is present, e.text must match via exact substring or
           normalized text matching.
         - If passage_id / sentence_idx is present, it must match e.evidence_id.
         - A document being retrieved does NOT automatically imply all of its
           supporting sentences were retrieved unless the context actually contains them.
         - If the retrieved item is explicitly marked as full document (is_full_document=True)
           and no text breakdown is provided, the full document context is assumed.

5. Evidence Completeness@K:
   Formula:
     Binary per-query indicator: 1.0 if EvidenceRecall@K == 1.0 (all required gold
     evidence units are covered by top-K retrieved passages), else 0.0.
   Dataset metric:
     Macro-average = fraction of evaluable questions for which every required evidence
     unit is covered by retrieved passages.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Set, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceItem:
    """Represents a discrete atomic unit of evidence (e.g., a supporting sentence or fact).

    Attributes:
        doc_id: The document identifier where this evidence unit resides.
        evidence_id: An identifier within the document (e.g., sentence index, span id,
            or fact label).
        text: Optional sentence/fact text string used for exact or normalized matching.
    """

    doc_id: str
    evidence_id: Union[str, int]
    text: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.doc_id).strip():
            raise ValueError("EvidenceItem doc_id must be a non-empty string.")


@dataclass(frozen=True)
class RetrievedPassage:
    """Represents a retrieved passage, sentence, or chunk candidate.

    Attributes:
        doc_id: The parent document identifier.
        passage_id: Optional unique passage or sentence identifier within the document.
        text: Optional passage text content.
        score: Retrieval score.
        rank: 1-indexed position in the ranked candidate list.
        is_full_document: True if this candidate represents the entire parent document,
            meaning all sentences of the document are present in the retrieval context.
    """

    doc_id: str
    passage_id: Optional[str] = None
    text: Optional[str] = None
    score: float = 0.0
    rank: int = 1
    is_full_document: bool = False

    def __post_init__(self) -> None:
        if not str(self.doc_id).strip():
            raise ValueError("RetrievedPassage doc_id must be a non-empty string.")
        if self.rank < 1:
            raise ValueError(f"Rank must be 1-indexed (>= 1), got rank={self.rank}.")


RetrievedInput = Union[str, RetrievedPassage, Dict[str, Any]]


@dataclass
class QueryResult:
    """Retrieved results for a single query."""

    query_id: str
    retrieved_doc_ids: List[str]


@dataclass
class QueryEvaluation:
    """Evaluation metrics and passage mappings for a single query."""

    query_id: str
    metrics: Dict[str, float] = field(default_factory=dict)
    doc_to_passages: Dict[str, List[RetrievedPassage]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Passage-to-Document Mapping & Relationship Preservation
# ---------------------------------------------------------------------------

def _to_retrieved_passage(item: RetrievedInput, rank: int) -> RetrievedPassage:
    """Coerce a retrieved item (string doc_id, dict, or RetrievedPassage) into a RetrievedPassage."""
    if isinstance(item, RetrievedPassage):
        return item
    if isinstance(item, dict):
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id:
            raise ValueError(f"Malformed passage dict at rank {rank}: missing or empty 'doc_id'.")
        return RetrievedPassage(
            doc_id=doc_id,
            passage_id=item.get("passage_id"),
            text=item.get("text"),
            score=float(item.get("score", 0.0)),
            rank=int(item.get("rank", rank)),
            is_full_document=bool(item.get("is_full_document", False)),
        )
    if isinstance(item, str):
        doc_id = item.strip()
        if not doc_id:
            raise ValueError(f"Malformed ranking: empty doc_id found at rank {rank}.")
        # A plain document ID is treated as a full document retrieval candidate
        return RetrievedPassage(
            doc_id=doc_id,
            rank=rank,
            is_full_document=True,
        )
    raise TypeError(f"Unsupported retrieved item type at rank {rank}: {type(item)}")


def map_passages_to_documents(
    passages: Sequence[RetrievedInput],
) -> List[str]:
    """Map retrieved passages/sentences back to parent document IDs for document metrics.

    Preserves rank order: the first time a parent document appears among the retrieved
    passages determines its rank in the document-level candidate list. Subsequent
    passages from the same document are deduplicated.

    Args:
        passages: Sequence of retrieved items (RetrievedPassage, dict, or string doc_ids).

    Returns:
        List of unique parent document IDs ordered by first appearance.
    """
    seen: Set[str] = set()
    doc_ids: List[str] = []
    for rank, item in enumerate(passages, start=1):
        p = _to_retrieved_passage(item, rank=rank)
        if p.doc_id not in seen:
            seen.add(p.doc_id)
            doc_ids.append(p.doc_id)
    return doc_ids


def preserve_doc_to_passages(
    passages: Sequence[RetrievedInput],
) -> Dict[str, List[RetrievedPassage]]:
    """Group retrieved passages by parent document ID while preserving rank order.

    Args:
        passages: Sequence of retrieved items.

    Returns:
        Dict mapping parent doc_id to the list of RetrievedPassage objects belonging to it.
    """
    doc_map: Dict[str, List[RetrievedPassage]] = {}
    for rank, item in enumerate(passages, start=1):
        p = _to_retrieved_passage(item, rank=rank)
        if p.doc_id not in doc_map:
            doc_map[p.doc_id] = []
        doc_map[p.doc_id].append(p)
    return doc_map


# ---------------------------------------------------------------------------
# Exact and Normalized Evidence Text Matching
# ---------------------------------------------------------------------------

def normalize_evidence_text(text: str) -> str:
    """Normalize text for robust exact or normalized evidence sentence matching.

    Normalizes unicode (NFKD), strips non-printable control characters, standardizes
    dashes/hyphens, removes enclosing quotes, collapses whitespace, and lowercases.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t ")
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uff0d\ufffd]", "-", text)
    text = re.sub(r"[\u2018\u2019\u201a\u201b\u2032\'\"]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def matches_evidence_text(evidence_text: str, context_text: str) -> bool:
    """Check whether evidence text appears in the context text.

    Uses exact substring match first, falling back to normalized text matching,
    punctuation-stripped matching, and exact token sequence matching.
    Never uses semantic similarity invented after retrieval.
    """
    if not evidence_text or not context_text:
        return False
    # 1. Exact substring match
    if evidence_text in context_text:
        return True
    # 2. Normalized text match
    norm_evidence = normalize_evidence_text(evidence_text)
    norm_context = normalize_evidence_text(context_text)
    if norm_evidence and norm_evidence in norm_context:
        return True
    # 3. Strip trailing sentence punctuation from evidence
    norm_ev_stripped = norm_evidence.rstrip(".!?,;: ")
    if norm_ev_stripped and norm_ev_stripped in norm_context:
        return True
    # 4. Exact token sequence matching (handles punctuation/tokenization variations)
    ev_tokens = re.findall(r"\b\w+\b", norm_evidence)
    if not ev_tokens:
        return False
    ctx_tokens = re.findall(r"\b\w+\b", norm_context)
    k = len(ev_tokens)
    for i in range(max(1, len(ctx_tokens) - k + 1)):
        if ctx_tokens[i : i + k] == ev_tokens:
            return True
    return False


def is_evidence_covered_by_passage(
    evidence: EvidenceItem,
    passage: RetrievedPassage,
) -> bool:
    """Determine whether an evidence unit is covered by a specific retrieved passage.

    Key rule: A document being retrieved does NOT automatically imply all of its
    supporting sentences were retrieved unless the retrieval context actually includes
    those sentences.

    Rules:
    1. Parent document ID must match (passage.doc_id == evidence.doc_id).
    2. If passage represents a full document (passage.is_full_document=True):
       - If passage.text and evidence.text are both provided, checks text inclusion.
       - If passage.text is None (representing document-level retrieval of the full document),
         the full document context is present, covering the evidence sentence.
    3. If passage represents a passage/sentence chunk (passage.is_full_document=False):
       - If passage.text and evidence.text are both provided:
         Must match via exact or normalized text matching. If passage.text does NOT
         contain evidence.text, returns False even though doc_id matches!
       - If passage.passage_id and evidence.evidence_id are both provided:
         Matches if str(passage.passage_id) == str(evidence.evidence_id).
       - If passage has text but does not match, returns False.
       - If neither text nor passage_id matches, returns False.
    """
    if str(passage.doc_id) != str(evidence.doc_id):
        return False

    # Full document retrieval mode
    if passage.is_full_document:
        if passage.text and evidence.text:
            return matches_evidence_text(evidence.text, passage.text)
        # Full document abstract retrieved with no text breakdown: entire document is present
        return True

    # Passage/chunk retrieval mode: context must actually include the sentence
    if passage.text and evidence.text:
        return matches_evidence_text(evidence.text, passage.text)

    if passage.passage_id is not None and evidence.evidence_id is not None:
        return str(passage.passage_id) == str(evidence.evidence_id)

    return False


def get_covered_evidence(
    retrieved: Sequence[RetrievedInput],
    gold_evidence: Collection[EvidenceItem],
    k: int,
) -> Set[EvidenceItem]:
    """Identify which gold evidence items are covered by the top-K retrieved passages."""
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got k={k}.")
    top_k_passages = [_to_retrieved_passage(item, rank=idx) for idx, item in enumerate(retrieved[:k], start=1)]
    covered: Set[EvidenceItem] = set()

    for ev in gold_evidence:
        for passage in top_k_passages:
            if is_evidence_covered_by_passage(ev, passage):
                covered.add(ev)
                break
    return covered


# ---------------------------------------------------------------------------
# Validation Utilities
# ---------------------------------------------------------------------------

def _validate_ranking(retrieved_docs: Sequence[str], allow_duplicates: bool = False) -> None:
    """Ensure retrieved ranking does not contain empty identifiers or unexpected duplicates."""
    seen: Set[str] = set()
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if not str(doc_id).strip():
            raise ValueError(f"Malformed ranking: empty doc_id found at rank {rank}.")
        if not allow_duplicates:
            if doc_id in seen:
                raise ValueError(
                    f"Malformed ranking: duplicate doc_id '{doc_id}' found at rank {rank}."
                )
            seen.add(doc_id)


# ---------------------------------------------------------------------------
# Document-Level Metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_docs: Sequence[str],
    relevant_docs: Collection[str],
    k: int,
    allow_duplicates: bool = False,
    empty_relevant_handling: str = "raise",
) -> float:
    r"""Compute conventional document-level Recall at cutoff rank K.

    Definition:
        Recall@K = \frac{|R@K \cap D^*|}{|D^*|}
    where:
        - R@K is the set of top-K retrieved document IDs.
        - D^* is the set of gold relevant document IDs.

    Denominator Choice:
        |D^*| (total count of relevant documents for this query).

    Edge Cases:
        - When |D^*| == 0:
            If empty_relevant_handling == "raise": raises ValueError (strict input validation).
            If empty_relevant_handling == "zero": returns 0.0.
        - When retrieved_docs is empty: returns 0.0.

    Args:
        retrieved_docs: Ordered sequence of retrieved document IDs (rank 1 to N).
        relevant_docs: Set or collection of gold relevant document IDs (D^*).
        k: Cutoff rank (must be > 0).
        allow_duplicates: If False, raises ValueError on duplicate doc IDs.
        empty_relevant_handling: "raise" or "zero" when relevant_docs is empty.

    Returns:
        float: Recall@K in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got k={k}.")
    _validate_ranking(retrieved_docs, allow_duplicates=allow_duplicates)

    rel_set = set(relevant_docs)
    if not rel_set:
        if empty_relevant_handling == "zero":
            return 0.0
        raise ValueError("relevant_docs cannot be empty; cannot compute Recall@K.")

    top_k_set = set(retrieved_docs[:k])
    hits = len(top_k_set & rel_set)
    return float(hits / len(rel_set))


def precision_at_k(
    retrieved_docs: Sequence[str],
    relevant_docs: Collection[str],
    k: int,
    allow_duplicates: bool = False,
    empty_relevant_handling: str = "raise",
) -> float:
    r"""Compute conventional document-level Precision at cutoff rank K.

    Definition:
        Precision@K = \frac{|R@K \cap D^*|}{K}
    where:
        - R@K is the set of top-K retrieved document IDs.
        - D^* is the set of gold relevant document IDs.
        - K is the fixed cutoff rank (standard IR definition).

    Denominator Choice & Behavior When Fewer Than K Documents Are Available:
        The denominator is strictly K. If fewer than K documents are retrieved
        (i.e. |retrieved_docs| < K), unretrieved slots (K - |retrieved_docs|) are
        treated as non-relevant documents. The denominator remains K.
        Example: If 2 relevant documents are retrieved and K=5, Precision@5 = 2/5 = 0.4.

    Args:
        retrieved_docs: Ordered sequence of retrieved document IDs (rank 1 to N).
        relevant_docs: Set or collection of gold relevant document IDs (D^*).
        k: Cutoff rank (must be > 0).
        allow_duplicates: If False, raises ValueError on duplicate doc IDs.
        empty_relevant_handling: "raise" or "zero" when relevant_docs is empty.

    Returns:
        float: Precision@K in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got k={k}.")
    _validate_ranking(retrieved_docs, allow_duplicates=allow_duplicates)

    rel_set = set(relevant_docs)
    if not rel_set:
        if empty_relevant_handling == "zero":
            return 0.0
        raise ValueError("relevant_docs cannot be empty; cannot compute Precision@K.")

    top_k_set = set(retrieved_docs[:k])
    hits = len(top_k_set & rel_set)
    return float(hits / k)


def reciprocal_rank(
    retrieved_docs: Sequence[str],
    relevant_docs: Collection[str],
    cutoff: Optional[int] = None,
    allow_duplicates: bool = False,
    empty_relevant_handling: str = "raise",
) -> float:
    r"""Compute Reciprocal Rank (RR) of the first relevant document.

    Definition:
        RR = 1 / rank_1 where rank_1 is the 1-indexed rank of the first relevant document.
        Queries with no relevant result in retrieved_docs (or within cutoff) receive 0.0.

    Args:
        retrieved_docs: Ordered sequence of retrieved document IDs.
        relevant_docs: Set or collection of gold relevant document IDs.
        cutoff: Optional maximum rank cutoff. If provided, hits beyond cutoff are ignored.
        allow_duplicates: If False, raises ValueError on duplicate doc IDs.
        empty_relevant_handling: "raise" or "zero" when relevant_docs is empty.

    Returns:
        float: Reciprocal rank in [0.0, 1.0].
    """
    if cutoff is not None and cutoff <= 0:
        raise ValueError(f"cutoff must be a positive integer, got cutoff={cutoff}.")
    _validate_ranking(retrieved_docs, allow_duplicates=allow_duplicates)

    rel_set = set(relevant_docs)
    if not rel_set:
        if empty_relevant_handling == "zero":
            return 0.0
        raise ValueError("relevant_docs cannot be empty; cannot compute Reciprocal Rank.")

    candidates = retrieved_docs if cutoff is None else retrieved_docs[:cutoff]
    for rank, doc_id in enumerate(candidates, start=1):
        if doc_id in rel_set:
            return float(1.0 / rank)
    return 0.0


# ---------------------------------------------------------------------------
# Evidence-Level Metrics
# ---------------------------------------------------------------------------

def evidence_recall_at_k(
    retrieved: Sequence[RetrievedInput],
    gold_evidence: Collection[EvidenceItem],
    k: int,
    empty_evidence_handling: str = "raise",
) -> float:
    r"""Compute Evidence Recall at cutoff rank K.

    Definition:
        Evidence Recall@K = |E_retrieved@K| / |E*|
    where:
        - E* is the set of annotated gold supporting evidence units.
        - E_retrieved@K is the set of gold evidence units covered by top-K retrieved passages.

    Denominator Choice:
        |E*| (total count of annotated gold evidence units for the query).

    Important Coverage Semantics:
        A document being retrieved does NOT automatically imply all of its supporting
        sentences were retrieved unless the retrieval context actually includes those
        sentences (checked via exact or normalized text matching, or passage_id matching).

    Args:
        retrieved: Ordered sequence of retrieved items (passages, dicts, or doc_ids).
        gold_evidence: Collection of gold EvidenceItem objects.
        k: Cutoff rank (must be > 0).
        empty_evidence_handling: "raise" or "zero" when gold_evidence is empty.

    Returns:
        float: Evidence Recall@K in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got k={k}.")

    ev_list = list(gold_evidence)
    if not ev_list:
        if empty_evidence_handling == "zero":
            return 0.0
        raise ValueError("gold_evidence cannot be empty; cannot compute Evidence Recall@K.")

    covered = get_covered_evidence(retrieved, ev_list, k=k)
    return float(len(covered) / len(ev_list))


def evidence_completeness_at_k(
    retrieved: Sequence[RetrievedInput],
    gold_evidence: Collection[EvidenceItem],
    k: int,
    empty_evidence_handling: str = "raise",
) -> float:
    r"""Compute Evidence Completeness at cutoff rank K.

    Definition:
        Evidence Completeness@K is a strict indicator (1.0 or 0.0) of whether ALL required
        gold evidence units are covered within the top-K retrieved passages:
            EvidenceCompleteness@K = 1.0 if EvidenceRecall@K == 1.0 else 0.0

    When aggregated across queries, this represents the fraction of evaluable questions
    for which every required evidence unit is covered by retrieved passages.

    Args:
        retrieved: Ordered sequence of retrieved items.
        gold_evidence: Collection of gold EvidenceItem objects.
        k: Cutoff rank (must be > 0).
        empty_evidence_handling: "raise" or "zero" when gold_evidence is empty.

    Returns:
        float: 1.0 if all evidence items are covered within top-K, 0.0 otherwise.
    """
    rec = evidence_recall_at_k(
        retrieved=retrieved,
        gold_evidence=gold_evidence,
        k=k,
        empty_evidence_handling=empty_evidence_handling,
    )
    return 1.0 if np.isclose(rec, 1.0) else 0.0


# ---------------------------------------------------------------------------
# Evaluation Pipeline & Aggregation
# ---------------------------------------------------------------------------

def evaluate_query(
    query_id: str,
    retrieved_docs: Sequence[RetrievedInput],
    relevant_docs: Collection[str],
    gold_evidence: Optional[Collection[EvidenceItem]] = None,
    cutoffs: Sequence[int] = (5, 10),
    empty_relevant_handling: str = "raise",
    empty_evidence_handling: str = "raise",
) -> QueryEvaluation:
    """Evaluate all document and evidence metrics for a single query across cutoffs.

    If retrieved_docs consists of passages or sentences, it automatically maps them
    back to parent document IDs for document-level metrics (Recall@K, Precision@K, MRR)
    while preserving document-to-passage mappings.

    Args:
        query_id: Unique query identifier.
        retrieved_docs: Ordered sequence of retrieved items (passages, dicts, or doc_ids).
        relevant_docs: Gold relevant document IDs.
        gold_evidence: Optional collection of gold EvidenceItem objects.
        cutoffs: Configurable sequence of rank cutoffs K (e.g. (5, 10)).
        empty_relevant_handling: "raise" or "zero" when relevant_docs is empty.
        empty_evidence_handling: "raise" or "zero" when gold_evidence is empty.

    Returns:
        QueryEvaluation with computed per-query metrics and preserved doc-to-passage mappings.
    """
    metrics: Dict[str, float] = {}

    # Map passages back to unique parent document IDs for document metrics
    doc_ids = map_passages_to_documents(retrieved_docs)
    doc_to_passages = preserve_doc_to_passages(retrieved_docs)

    # Document-level metrics
    for k in cutoffs:
        metrics[f"Recall@{k}"] = recall_at_k(
            doc_ids,
            relevant_docs,
            k=k,
            allow_duplicates=True,
            empty_relevant_handling=empty_relevant_handling,
        )
        metrics[f"Precision@{k}"] = precision_at_k(
            doc_ids,
            relevant_docs,
            k=k,
            allow_duplicates=True,
            empty_relevant_handling=empty_relevant_handling,
        )

    metrics["MRR"] = reciprocal_rank(
        doc_ids,
        relevant_docs,
        allow_duplicates=True,
        empty_relevant_handling=empty_relevant_handling,
    )

    # Evidence-level metrics (if evidence ground truth is provided)
    if gold_evidence is not None:
        for k in cutoffs:
            metrics[f"EvidenceRecall@{k}"] = evidence_recall_at_k(
                retrieved=retrieved_docs,
                gold_evidence=gold_evidence,
                k=k,
                empty_evidence_handling=empty_evidence_handling,
            )
            metrics[f"EvidenceCompleteness@{k}"] = evidence_completeness_at_k(
                retrieved=retrieved_docs,
                gold_evidence=gold_evidence,
                k=k,
                empty_evidence_handling=empty_evidence_handling,
            )

    return QueryEvaluation(
        query_id=query_id,
        metrics=metrics,
        doc_to_passages=doc_to_passages,
    )


def aggregate_evaluations(
    evaluations: Sequence[QueryEvaluation],
) -> Dict[str, float]:
    """Aggregate per-query evaluations into corpus-level macro-averaged metrics.

    Args:
        evaluations: Sequence of QueryEvaluation objects.

    Returns:
        Dict mapping metric name to macro-averaged mean value across queries.

    Raises:
        ValueError: If evaluations sequence is empty or inconsistent.
    """
    if not evaluations:
        raise ValueError("Cannot aggregate empty list of evaluations.")

    metric_keys = list(evaluations[0].metrics.keys())
    accumulators: Dict[str, List[float]] = {k: [] for k in metric_keys}

    for ev in evaluations:
        for k in metric_keys:
            if k not in ev.metrics:
                raise ValueError(
                    f"Query {ev.query_id} is missing metric '{k}'; cannot aggregate incomplete evaluations."
                )
            accumulators[k].append(ev.metrics[k])

    return {k: float(np.mean(vals)) for k, vals in accumulators.items()}
