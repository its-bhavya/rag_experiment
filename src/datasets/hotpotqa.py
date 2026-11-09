"""HotpotQA distractor-setting dataset loader and processor.

This module loads the HotpotQA distractor validation split from HuggingFace,
samples N=500 questions using a fixed seed, constructs per-question candidate
paragraph corpora from the supplied distractor paragraphs, maps supporting facts
to EvidenceItem objects, and saves processed data to disk.

IMPORTANT LIMITATION:
    The candidate corpus is constructed from the 10 distractor paragraphs
    supplied per question in the HotpotQA distractor setting.
    This is NOT full-Wikipedia retrieval.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path before local imports
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.metrics import EvidenceItem  # noqa: E402


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HotpotParagraph:
    """A single candidate paragraph for a HotpotQA question.

    Attributes:
        doc_id: Unique identifier ``f"{question_id}___{title}"``.
        question_id: Parent question identifier.
        title: Wikipedia article title.
        sentences: List of sentence strings.
        full_text: ``f"{title}\\n{' '.join(sentences)}"``.
    """

    doc_id: str
    question_id: str
    title: str
    sentences: List[str]
    full_text: str


@dataclass
class HotpotQuestion:
    """A fully processed HotpotQA question with candidate paragraphs and evidence.

    Attributes:
        question_id: Original ``_id`` field from HotpotQA.
        question: Question text.
        answer: Gold answer string.
        q_type: Question type (``"bridge"`` or ``"comparison"``).
        level: Difficulty level (``"easy"``, ``"medium"``, or ``"hard"``).
        supporting_facts: List of ``(title, sent_idx)`` pairs from the annotation.
        paragraphs: Mapping from ``doc_id`` to :class:`HotpotParagraph`.
        evidence_items: :class:`EvidenceItem` objects for valid supporting facts.
        invalid_supporting_facts: Flagged annotations that could not be resolved.
        gold_doc_ids: ``doc_id`` values for paragraphs that contain at least one
            supporting fact.
    """

    question_id: str
    question: str
    answer: str
    q_type: str
    level: str
    supporting_facts: List[Tuple[str, int]]
    paragraphs: Dict[str, HotpotParagraph]
    evidence_items: List[EvidenceItem]
    invalid_supporting_facts: List[Dict[str, Any]]
    gold_doc_ids: List[str]


@dataclass
class HotpotDataset:
    """Collection of processed HotpotQA questions plus sampling metadata.

    Attributes:
        questions: Mapping from ``question_id`` to :class:`HotpotQuestion`.
        sampling_seed: Random seed used for sampling.
        sample_size: Number of questions sampled.
        metadata: Provenance and construction details.
    """

    questions: Dict[str, HotpotQuestion]
    sampling_seed: int
    sample_size: int
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_paragraphs(
    question_id: str,
    context: List[List],
) -> Dict[str, HotpotParagraph]:
    """Build a ``doc_id -> HotpotParagraph`` mapping from a question's context field."""
    paragraphs: Dict[str, HotpotParagraph] = {}
    for entry in context:
        # Each entry is [title, list_of_sentences]
        title: str = entry[0]
        sentences: List[str] = list(entry[1])
        doc_id = f"{question_id}___{title}"
        full_text = f"{title}\n{' '.join(sentences)}"
        paragraphs[doc_id] = HotpotParagraph(
            doc_id=doc_id,
            question_id=question_id,
            title=title,
            sentences=sentences,
            full_text=full_text,
        )
    return paragraphs


def _map_supporting_facts(
    question_id: str,
    supporting_facts: List[List],
    paragraphs: Dict[str, HotpotParagraph],
) -> Tuple[List[EvidenceItem], List[Dict[str, Any]]]:
    """Resolve each supporting fact to an :class:`EvidenceItem`.

    Returns
    -------
    evidence_items:
        Successfully resolved evidence items.
    invalid_facts:
        Dicts describing facts that could not be resolved (title missing or
        sent_idx out of range).
    """
    # Build a title -> paragraph lookup for this question
    title_to_para: Dict[str, HotpotParagraph] = {
        p.title: p for p in paragraphs.values()
    }

    evidence_items: List[EvidenceItem] = []
    invalid_facts: List[Dict[str, Any]] = []

    for fact in supporting_facts:
        title: str = fact[0]
        sent_idx: int = int(fact[1])

        if title not in title_to_para:
            invalid_facts.append(
                {
                    "question_id": question_id,
                    "title": title,
                    "sent_idx": sent_idx,
                    "reason": "title_not_in_context",
                }
            )
            logger.debug(
                "Invalid supporting fact for question %s: title '%s' not found in context",
                question_id,
                title,
            )
            continue

        para = title_to_para[title]
        if sent_idx < 0 or sent_idx >= len(para.sentences):
            invalid_facts.append(
                {
                    "question_id": question_id,
                    "title": title,
                    "sent_idx": sent_idx,
                    "reason": "sent_idx_out_of_range",
                    "num_sentences": len(para.sentences),
                }
            )
            logger.debug(
                "Invalid supporting fact for question %s: sent_idx %d out of range "
                "for title '%s' (has %d sentences)",
                question_id,
                sent_idx,
                title,
                len(para.sentences),
            )
            continue

        sentence_text = para.sentences[sent_idx]
        doc_id = para.doc_id
        evidence_items.append(
            EvidenceItem(doc_id=doc_id, evidence_id=sent_idx, text=sentence_text)
        )

    return evidence_items, invalid_facts


def _build_gold_doc_ids(evidence_items: List[EvidenceItem]) -> List[str]:
    """Return unique doc_ids referenced by the evidence items (preserving order)."""
    seen: set = set()
    result: List[str] = []
    for ev in evidence_items:
        if ev.doc_id not in seen:
            seen.add(ev.doc_id)
            result.append(ev.doc_id)
    return result


def _process_raw_question(raw: Dict[str, Any]) -> HotpotQuestion:
    """Convert a raw HuggingFace HotpotQA record into a :class:`HotpotQuestion`."""
    question_id: str = str(raw["id"])
    question_text: str = raw["question"]
    answer: str = raw["answer"]
    q_type: str = raw["type"]
    level: str = raw["level"]

    # supporting_facts is a dict: {"title": [...], "sent_id": [...]}
    sf_dict = raw["supporting_facts"]
    supporting_facts: List[Tuple[str, int]] = list(
        zip(sf_dict["title"], sf_dict["sent_id"])
    )

    # context is a dict: {"title": [...], "sentences": [[...], ...]}
    ctx_dict = raw["context"]
    context_list = list(zip(ctx_dict["title"], ctx_dict["sentences"]))

    paragraphs = _build_paragraphs(question_id, context_list)
    evidence_items, invalid_facts = _map_supporting_facts(
        question_id, supporting_facts, paragraphs
    )
    gold_doc_ids = _build_gold_doc_ids(evidence_items)

    return HotpotQuestion(
        question_id=question_id,
        question=question_text,
        answer=answer,
        q_type=q_type,
        level=level,
        supporting_facts=supporting_facts,
        paragraphs=paragraphs,
        evidence_items=evidence_items,
        invalid_supporting_facts=invalid_facts,
        gold_doc_ids=gold_doc_ids,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_hotpotqa_distractor(
    sample_size: int = 500,
    seed: int = 12345,
) -> HotpotDataset:
    """Load and sample the HotpotQA distractor validation split.

    Parameters
    ----------
    sample_size:
        Number of questions to sample.  Defaults to 500.
    seed:
        Random seed for reproducible sampling.  Defaults to 12345.

    Returns
    -------
    HotpotDataset
        Fully processed dataset with per-question candidate corpora,
        evidence items, and invalid-fact flags.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' library is required. Install it with: pip install datasets"
        ) from exc

    logger.info(
        "Loading HotpotQA distractor validation split from HuggingFace..."
    )
    hf_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    total_available = len(hf_dataset)
    logger.info("Total available questions: %d", total_available)

    # Reproducible sampling — ONLY by fixed seed, no other criterion
    all_indices = list(range(total_available))
    rng = random.Random(seed)
    sampled_indices = rng.sample(all_indices, min(sample_size, total_available))
    logger.info(
        "Sampled %d questions with seed=%d", len(sampled_indices), seed
    )

    questions: Dict[str, HotpotQuestion] = {}
    total_invalid = 0
    for idx in sampled_indices:
        raw = hf_dataset[idx]
        q = _process_raw_question(raw)
        questions[q.question_id] = q
        total_invalid += len(q.invalid_supporting_facts)

    total_paragraphs = sum(len(q.paragraphs) for q in questions.values())
    total_evidence = sum(len(q.evidence_items) for q in questions.values())

    metadata: Dict[str, Any] = {
        "sampling_seed": seed,
        "sample_size": len(questions),
        "hf_dataset_id": "hotpotqa/hotpot_qa",
        "hf_config": "distractor",
        "hf_split": "validation",
        "construction_method": (
            "Candidate corpus constructed from distractor paragraphs supplied per question. "
            "This is NOT full-Wikipedia retrieval."
        ),
        "total_questions": len(questions),
        "total_paragraphs": total_paragraphs,
        "total_evidence_items": total_evidence,
        "invalid_supporting_facts_count": total_invalid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "Processed %d questions | %d paragraphs | %d evidence items | "
        "%d invalid supporting facts",
        len(questions),
        total_paragraphs,
        total_evidence,
        total_invalid,
    )
    logger.warning(
        "LIMITATION: %s",
        metadata["construction_method"],
    )

    return HotpotDataset(
        questions=questions,
        sampling_seed=seed,
        sample_size=len(questions),
        metadata=metadata,
    )


def save_processed_data(
    dataset: HotpotDataset,
    output_dir: Path,
) -> None:
    """Persist processed HotpotQA data to disk.

    Saves four files:

    * ``hotpotqa_questions.jsonl`` – one line per question.
    * ``hotpotqa_candidates.jsonl`` – one line per candidate paragraph.
    * ``hotpotqa_evidence.jsonl`` – one line per evidence item.
    * ``hotpotqa_metadata.json`` – sampling and provenance metadata.

    Parameters
    ----------
    dataset:
        Processed :class:`HotpotDataset`.
    output_dir:
        Directory to write output files into (created if necessary).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- questions ---
    questions_path = output_dir / "hotpotqa_questions.jsonl"
    with open(questions_path, "w", encoding="utf-8") as f:
        for q in dataset.questions.values():
            record = {
                "question_id": q.question_id,
                "question": q.question,
                "answer": q.answer,
                "type": q.q_type,
                "level": q.level,
                "supporting_facts": [
                    {"title": t, "sent_idx": s} for t, s in q.supporting_facts
                ],
                "gold_doc_ids": q.gold_doc_ids,
                "invalid_supporting_facts": q.invalid_supporting_facts,
                "evidence_item_count": len(q.evidence_items),
                "paragraph_count": len(q.paragraphs),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d questions to %s", len(dataset.questions), questions_path)

    # --- candidates ---
    candidates_path = output_dir / "hotpotqa_candidates.jsonl"
    total_paras = 0
    with open(candidates_path, "w", encoding="utf-8") as f:
        for q in dataset.questions.values():
            for para in q.paragraphs.values():
                record = {
                    "doc_id": para.doc_id,
                    "question_id": para.question_id,
                    "title": para.title,
                    "sentences": para.sentences,
                    "full_text": para.full_text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_paras += 1
    logger.info("Saved %d candidate paragraphs to %s", total_paras, candidates_path)

    # --- evidence ---
    evidence_path = output_dir / "hotpotqa_evidence.jsonl"
    total_ev = 0
    with open(evidence_path, "w", encoding="utf-8") as f:
        for q in dataset.questions.values():
            for ev in q.evidence_items:
                # Determine match status
                para = q.paragraphs.get(ev.doc_id)
                if para is not None and isinstance(ev.evidence_id, int):
                    if 0 <= ev.evidence_id < len(para.sentences):
                        actual_text = para.sentences[ev.evidence_id]
                        match_status = "exact" if actual_text == ev.text else "mismatch"
                        match_confidence = 1.0 if match_status == "exact" else 0.0
                    else:
                        match_status = "index_error"
                        match_confidence = 0.0
                else:
                    match_status = "unknown"
                    match_confidence = 0.0

                record = {
                    "query_id": q.question_id,
                    "evidence_doc_id": ev.doc_id,
                    "sentence_idx": ev.evidence_id,
                    "evidence_sentence": ev.text,
                    "evidence_match_status": match_status,
                    "match_method": "sentence_index",
                    "match_confidence": match_confidence,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_ev += 1
    logger.info("Saved %d evidence items to %s", total_ev, evidence_path)

    # --- metadata ---
    metadata_path = output_dir / "hotpotqa_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(dataset.metadata, f, indent=2, ensure_ascii=False)
    logger.info("Saved metadata to %s", metadata_path)


# ---------------------------------------------------------------------------
# CLI entry-point for data download/processing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Download and process HotpotQA distractor validation split."
    )
    parser.add_argument(
        "--sample-size", type=int, default=500, help="Number of questions to sample."
    )
    parser.add_argument(
        "--seed", type=int, default=12345, help="Random seed for sampling."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "data" / "processed",
        help="Output directory for processed files.",
    )
    args = parser.parse_args()

    dataset = load_hotpotqa_distractor(
        sample_size=args.sample_size,
        seed=args.seed,
    )
    save_processed_data(dataset, args.output_dir)
    print(
        f"\nDone. Saved {dataset.sample_size} questions to {args.output_dir}"
    )
