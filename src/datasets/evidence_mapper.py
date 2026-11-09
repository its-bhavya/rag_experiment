"""Evidence mapping and normalization module for SciFact.

Maps claims and supporting sentence evidence from official SciFact annotations
to BEIR corpus documents using robust hierarchical normalized text matching.
Preserves:
- query_id
- evidence_doc_id
- evidence_sentence
- evidence_annotation_id
- evidence_match_status (matched | unmatched)
- match_method (exact | normalized | token_overlap | unmatched)
- match_confidence (float in [0.0, 1.0])
- unmapped_reason (if unmatched)
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.evaluation.metrics import EvidenceItem

logger = logging.getLogger(__name__)


@dataclass
class EvidenceMappingRecord:
    """Represents a mapped or unmapped evidence annotation."""

    query_id: str
    evidence_doc_id: str
    sentence_idx: int
    evidence_sentence: str
    evidence_annotation_id: str
    evidence_match_status: str  # "matched" or "unmatched"
    match_method: str  # "exact", "normalized", "token_overlap", "unmatched"
    match_confidence: float
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    unmapped_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_for_matching(text: str) -> str:
    """Normalize text for robust matching across unicode, punctuation, and whitespace variations."""
    if not text:
        return ""
    # Unicode NFKD normalization
    text = unicodedata.normalize("NFKD", text)
    # Remove control characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t ")
    # Normalize varied dashes, hyphens, and minus signs to standard hyphen
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uff0d\ufffd]", "-", text)
    # Normalize curly/directional single and double quotes
    text = re.sub(r"[\u2018\u2019\u201a\u201b\u2032]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e\u201f\u2033]", '"', text)
    # Collapse arbitrary whitespace (newlines, tabs, multiple spaces) into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_tokens(text: str) -> List[str]:
    """Extract lowercased alphanumeric tokens for fallback fuzzy/token matching."""
    norm = normalize_for_matching(text).lower()
    return re.findall(r"\b\w+\b", norm)


def match_sentence_in_document(
    sentence: str,
    doc_text: str,
    min_overlap_threshold: float = 0.85,
) -> Tuple[bool, str, float, Optional[int], Optional[int], Optional[str]]:
    """Match a supporting sentence against a corpus document text.

    Uses a hierarchical strategy:
    1. Exact literal substring match
    2. Normalized substring match (unicode + punctuation + whitespace collapsed)
    3. Token overlap fallback (for minor OCR or hyphenation differences)

    Returns:
        (is_matched, match_method, confidence, span_start, span_end, unmapped_reason)
    """
    if not sentence.strip():
        return False, "unmatched", 0.0, None, None, "empty_evidence_sentence"
    if not doc_text.strip():
        return False, "unmatched", 0.0, None, None, "empty_document_text"

    # Step 1: Exact literal match
    idx = doc_text.find(sentence)
    if idx != -1:
        return True, "exact", 1.0, idx, idx + len(sentence), None

    # Step 2: Normalized match
    norm_sent = normalize_for_matching(sentence)
    norm_doc = normalize_for_matching(doc_text)

    idx_norm = norm_doc.find(norm_sent)
    if idx_norm != -1:
        return True, "normalized", 0.98, idx_norm, idx_norm + len(norm_sent), None

    # Step 2b: Try matching without section headers that SciFact abstracts sometimes embed
    # e.g., sentence starting with or containing "CONCLUSION ", "BACKGROUND ", etc.
    header_pattern = r"^(BACKGROUND|METHODS|RESULTS|CONCLUSIONS?|CONTEXT|OBJECTIVE|UNLABELLED)\s+"
    stripped_sent = re.sub(header_pattern, "", norm_sent, flags=re.IGNORECASE).strip()
    if stripped_sent and len(stripped_sent) > 15:
        idx_stripped = norm_doc.find(stripped_sent)
        if idx_stripped != -1:
            return True, "normalized_stripped_header", 0.95, idx_stripped, idx_stripped + len(stripped_sent), None

    # Step 3: Token sequence matching (handles tokenization / hyphenation splits)
    sent_tokens = clean_tokens(norm_sent)
    if not sent_tokens:
        return False, "unmatched", 0.0, None, None, "no_valid_tokens_in_evidence"

    doc_tokens = clean_tokens(norm_doc)
    doc_token_set = set(doc_tokens)
    overlap = sum(1 for t in sent_tokens if t in doc_token_set)
    overlap_ratio = overlap / len(sent_tokens)

    # Check if a sliding window of tokens matches closely
    k = len(sent_tokens)
    best_window_score = 0.0
    for i in range(max(1, len(doc_tokens) - k + 1)):
        window = doc_tokens[i : i + k]
        matches = sum(1 for a, b in zip(sent_tokens, window) if a == b)
        score = matches / k
        if score > best_window_score:
            best_window_score = score
            if best_window_score >= 0.95:
                break

    effective_score = max(overlap_ratio, best_window_score)
    if effective_score >= min_overlap_threshold:
        return True, "token_overlap", float(round(effective_score, 4)), None, None, None

    return (
        False,
        "unmatched",
        float(round(effective_score, 4)),
        None,
        None,
        f"low_similarity_overlap_{effective_score:.2f}",
    )


def build_evidence_mappings(
    claims_path: Path | str,
    official_corpus_path: Path | str,
    beir_corpus_path: Path | str,
) -> Tuple[List[EvidenceMappingRecord], Dict[str, Any]]:
    """Build reliable evidence mappings from official SciFact files to BEIR corpus.

    Args:
        claims_path: Path to official claims JSONL (e.g. claims_dev.jsonl).
        official_corpus_path: Path to official corpus JSONL with sentence-segmented abstracts.
        beir_corpus_path: Path to BEIR corpus JSONL.

    Returns:
        Tuple of (list of EvidenceMappingRecord, mapping_report_dictionary).
    """
    # 1. Load official corpus (doc_id -> abstract list)
    official_corpus: Dict[str, List[str]] = {}
    with open(official_corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            doc_id = str(item["doc_id"])
            official_corpus[doc_id] = item.get("abstract", [])

    # 2. Load BEIR corpus (doc_id -> full text)
    beir_corpus: Dict[str, str] = {}
    with open(beir_corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            beir_corpus[str(item["_id"])] = item.get("text", "")

    # 3. Iterate over claims and extract annotated evidence
    records: List[EvidenceMappingRecord] = []
    claims_total = 0
    claims_with_evidence = 0

    # For reporting query completeness
    query_evidence_counts: Dict[str, int] = {}
    query_matched_counts: Dict[str, int] = {}

    with open(claims_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            claims_total += 1
            item = json.loads(line)
            qid = str(item["id"])
            evidence_dict = item.get("evidence", {})

            if evidence_dict:
                claims_with_evidence += 1

            for doc_id, ev_list in evidence_dict.items():
                doc_id_str = str(doc_id)
                abstract_sentences = official_corpus.get(doc_id_str, [])
                doc_text = beir_corpus.get(doc_id_str, "")

                for ev_group_idx, ev in enumerate(ev_list):
                    label = ev.get("label", "UNKNOWN")
                    sentences = ev.get("sentences", [])

                    for s_idx in sentences:
                        annotation_id = f"{qid}_{doc_id_str}_g{ev_group_idx}_s{s_idx}"
                        query_evidence_counts[qid] = query_evidence_counts.get(qid, 0) + 1

                        # Validate sentence index
                        if doc_id_str not in official_corpus:
                            rec = EvidenceMappingRecord(
                                query_id=qid,
                                evidence_doc_id=doc_id_str,
                                sentence_idx=s_idx,
                                evidence_sentence="",
                                evidence_annotation_id=annotation_id,
                                evidence_match_status="unmatched",
                                match_method="unmatched",
                                match_confidence=0.0,
                                unmapped_reason="document_not_in_official_corpus",
                                metadata={"label": label},
                            )
                        elif s_idx >= len(abstract_sentences):
                            rec = EvidenceMappingRecord(
                                query_id=qid,
                                evidence_doc_id=doc_id_str,
                                sentence_idx=s_idx,
                                evidence_sentence="",
                                evidence_annotation_id=annotation_id,
                                evidence_match_status="unmatched",
                                match_method="unmatched",
                                match_confidence=0.0,
                                unmapped_reason=f"sentence_index_{s_idx}_exceeds_abstract_len_{len(abstract_sentences)}",
                                metadata={"label": label},
                            )
                        elif doc_id_str not in beir_corpus:
                            raw_sentence = abstract_sentences[s_idx]
                            rec = EvidenceMappingRecord(
                                query_id=qid,
                                evidence_doc_id=doc_id_str,
                                sentence_idx=s_idx,
                                evidence_sentence=raw_sentence,
                                evidence_annotation_id=annotation_id,
                                evidence_match_status="unmatched",
                                match_method="unmatched",
                                match_confidence=0.0,
                                unmapped_reason="document_missing_from_beir_corpus",
                                metadata={"label": label},
                            )
                        else:
                            raw_sentence = abstract_sentences[s_idx]
                            matched, method, conf, sp_start, sp_end, reason = (
                                match_sentence_in_document(raw_sentence, doc_text)
                            )
                            status = "matched" if matched else "unmatched"
                            rec = EvidenceMappingRecord(
                                query_id=qid,
                                evidence_doc_id=doc_id_str,
                                sentence_idx=s_idx,
                                evidence_sentence=raw_sentence,
                                evidence_annotation_id=annotation_id,
                                evidence_match_status=status,
                                match_method=method,
                                match_confidence=conf,
                                span_start=sp_start,
                                span_end=sp_end,
                                unmapped_reason=reason,
                                metadata={"label": label},
                            )

                        if rec.evidence_match_status == "matched":
                            query_matched_counts[qid] = query_matched_counts.get(qid, 0) + 1

                        records.append(rec)

    # 4. Compute completeness statistics
    total_evidence_annotations = len(records)
    successfully_mapped = sum(1 for r in records if r.evidence_match_status == "matched")
    unmatched_evidence = total_evidence_annotations - successfully_mapped

    queries_with_complete_mappings: List[str] = []
    queries_with_partial_mappings: List[str] = []
    queries_with_zero_mappings: List[str] = []

    for qid, total_ev in query_evidence_counts.items():
        matched_ev = query_matched_counts.get(qid, 0)
        if matched_ev == total_ev:
            queries_with_complete_mappings.append(qid)
        elif matched_ev > 0:
            queries_with_partial_mappings.append(qid)
        else:
            queries_with_zero_mappings.append(qid)

    # Match method breakdown
    method_counts: Dict[str, int] = {}
    for r in records:
        method_counts[r.match_method] = method_counts.get(r.match_method, 0) + 1

    report = {
        "dataset": "SciFact",
        "split": "test",
        "source_claims": str(claims_path),
        "total_claims": claims_total,
        "claims_with_evidence": claims_with_evidence,
        "mapping_summary": {
            "total_evidence_annotations": total_evidence_annotations,
            "successfully_mapped_evidence": successfully_mapped,
            "unmatched_evidence": unmatched_evidence,
            "mapping_success_rate": (
                float(successfully_mapped / total_evidence_annotations)
                if total_evidence_annotations > 0
                else 0.0
            ),
        },
        "query_completeness": {
            "total_queries_with_evidence": len(query_evidence_counts),
            "queries_with_complete_evidence_mappings_count": len(queries_with_complete_mappings),
            "queries_with_complete_evidence_mappings": sorted(queries_with_complete_mappings, key=int),
            "queries_with_partial_evidence_mappings_count": len(queries_with_partial_mappings),
            "queries_with_partial_evidence_mappings": sorted(queries_with_partial_mappings, key=int),
            "queries_with_zero_evidence_mappings_count": len(queries_with_zero_mappings),
            "queries_with_zero_evidence_mappings": sorted(queries_with_zero_mappings, key=int),
        },
        "match_methods_breakdown": method_counts,
    }

    return records, report


def save_evidence_mappings(
    records: List[EvidenceMappingRecord],
    report: Dict[str, Any],
    output_jsonl: Path | str = "data/processed/scifact_evidence.jsonl",
    output_report: Path | str = "data/processed/evidence_mapping_report.json",
) -> None:
    """Save mapped evidence records to JSONL and diagnostic report to JSON."""
    out_jsonl = Path(output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    out_report = Path(output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved {len(records)} evidence mappings to {out_jsonl}")
    logger.info(f"Saved evidence mapping report to {out_report}")


def load_verified_evidence_ground_truth(
    evidence_jsonl: Path | str = "data/processed/scifact_evidence.jsonl",
    report_json: Path | str = "data/processed/evidence_mapping_report.json",
) -> Tuple[Dict[str, List[EvidenceItem]], Set[str]]:
    """Load verified evidence ground truth strictly restricted to complete mappings.

    Guarantees requirement 9:
    'Only evaluate evidence metrics on queries whose annotations meet the documented
    completeness requirements.'

    Returns:
        Tuple of (mapping from query_id to list of EvidenceItem, set of eligible query IDs).
    """
    with open(report_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    complete_query_ids = set(report["query_completeness"]["queries_with_complete_evidence_mappings"])

    evidence_by_query: Dict[str, List[EvidenceItem]] = {qid: [] for qid in complete_query_ids}
    with open(evidence_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = r["query_id"]
            if qid in complete_query_ids and r["evidence_match_status"] == "matched":
                evidence_by_query[qid].append(
                    EvidenceItem(
                        doc_id=r["evidence_doc_id"],
                        evidence_id=r["sentence_idx"],
                        text=r.get("evidence_sentence"),
                    )
                )

    return evidence_by_query, complete_query_ids
