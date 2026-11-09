"""SciFact dataset loader and processor.

Downloads, parses, validates, and processes the SciFact dataset from BEIR,
preserving all original IDs, texts, metadata, and supporting evidence annotations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import zipfile

from src.evaluation.metrics import EvidenceItem

logger = logging.getLogger(__name__)

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
EXPECTED_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"


@dataclass
class SciFactDocument:
    """Represents a document in the SciFact corpus."""

    id: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Combined title and body text."""
        return f"{self.title}\n{self.text}".strip() if self.title else self.text


@dataclass
class SciFactQuery:
    """Represents a claim/query in SciFact."""

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence: List[EvidenceItem] = field(default_factory=list)


@dataclass
class SciFactDataset:
    """Container for the parsed SciFact corpus, queries, and qrels."""

    corpus: Dict[str, SciFactDocument]
    queries: Dict[str, SciFactQuery]
    qrels: Dict[str, Dict[str, int]]  # query_id -> {doc_id: score}
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_sha256(file_path: Path | str) -> str:
    """Compute SHA256 checksum of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def download_scifact(
    raw_dir: Path | str = "data/raw",
    force: bool = False,
    url: str = SCIFACT_URL,
) -> Path:
    """Download SciFact zip archive and extract to data/raw/scifact/.

    Args:
        raw_dir: Root raw data directory.
        force: If True, re-download even if archive already exists.
        url: Source URL for the dataset zip.

    Returns:
        Path to extracted scifact directory.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    zip_path = raw_path / "scifact.zip"
    extract_target = raw_path / "scifact"

    if force or not zip_path.exists():
        logger.info(f"Downloading SciFact from {url} to {zip_path}...")
        try:
            import certifi
            import requests

            response = requests.get(url, stream=True, verify=certifi.where(), timeout=60)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            # Fallback to urllib if requests fails
            import urllib.request
            logger.warning(f"requests download failed ({e}); falling back to urllib.request...")
            urllib.request.urlretrieve(url, zip_path)

    checksum = compute_sha256(zip_path)
    logger.info(f"SciFact archive SHA256: {checksum}")

    # Extract archive if not already extracted or forced
    if force or not (extract_target / "corpus.jsonl").exists():
        logger.info(f"Extracting {zip_path} to {raw_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_path)

    # Save raw download metadata
    metadata = {
        "dataset": "scifact",
        "source_url": url,
        "archive_file": str(zip_path.name),
        "sha256": checksum,
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(extract_target / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return extract_target


def parse_corpus(corpus_path: Path | str) -> Dict[str, SciFactDocument]:
    """Parse BEIR format corpus.jsonl preserving original fields and IDs."""
    corpus: Dict[str, SciFactDocument] = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            doc_id = str(item["_id"])
            if doc_id in corpus:
                raise ValueError(f"Duplicate doc_id '{doc_id}' encountered at line {line_num}.")
            corpus[doc_id] = SciFactDocument(
                id=doc_id,
                title=item.get("title", ""),
                text=item.get("text", ""),
                metadata=item.get("metadata", {}),
            )
    return corpus


def extract_evidence_from_metadata(metadata: Dict[str, Any]) -> List[EvidenceItem]:
    """Extract discrete EvidenceItem instances from SciFact query metadata.

    In SciFact / BEIR, query metadata maps doc_id to a list of dicts with
    'sentences' (list of int sentence indices) and 'label' ('SUPPORT'/'CONTRADICT').
    """
    evidence_items: List[EvidenceItem] = []
    if not isinstance(metadata, dict):
        return evidence_items

    for doc_id, doc_evidence_list in metadata.items():
        if isinstance(doc_evidence_list, list):
            for ev in doc_evidence_list:
                if isinstance(ev, dict) and "sentences" in ev:
                    for sent_idx in ev["sentences"]:
                        evidence_items.append(
                            EvidenceItem(doc_id=str(doc_id), evidence_id=sent_idx)
                        )
    return evidence_items


def parse_queries(queries_path: Path | str) -> Dict[str, SciFactQuery]:
    """Parse queries.jsonl preserving original fields, IDs, and evidence annotations."""
    queries: Dict[str, SciFactQuery] = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            q_id = str(item["_id"])
            if q_id in queries:
                raise ValueError(f"Duplicate query_id '{q_id}' encountered at line {line_num}.")
            meta = item.get("metadata", {})
            evidence = extract_evidence_from_metadata(meta)
            queries[q_id] = SciFactQuery(
                id=q_id,
                text=item.get("text", ""),
                metadata=meta,
                evidence=evidence,
            )
    return queries


def parse_qrels(qrels_path: Path | str) -> Dict[str, Dict[str, int]]:
    """Parse BEIR qrels TSV file (query-id, corpus-id, score)."""
    qrels: Dict[str, Dict[str, int]] = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None or len(header) < 3:
            raise ValueError(f"Malformed or empty qrels file: {qrels_path}")

        for row_num, row in enumerate(reader, start=2):
            if not row or len(row) < 3:
                continue
            qid, docid, score_str = row[0].strip(), row[1].strip(), row[2].strip()
            if not qid or not docid:
                raise ValueError(f"Empty qid or docid at row {row_num} in {qrels_path}.")
            score = int(score_str)
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = score
    return qrels


def load_scifact_dataset(
    data_dir: Path | str = "data/raw/scifact",
    split: str = "test",
) -> SciFactDataset:
    """Load and parse the full SciFact dataset (corpus, queries, and qrels for specified split)."""
    base_dir = Path(data_dir)
    corpus_file = base_dir / "corpus.jsonl"
    queries_file = base_dir / "queries.jsonl"
    qrels_file = base_dir / "qrels" / f"{split}.tsv"

    if not corpus_file.exists() or not queries_file.exists() or not qrels_file.exists():
        raise FileNotFoundError(
            f"Missing required SciFact files in {base_dir}. Run download_scifact() first."
        )

    corpus = parse_corpus(corpus_file)
    queries = parse_queries(queries_file)
    qrels = parse_qrels(qrels_file)

    metadata_file = base_dir / "metadata.json"
    archive_meta = {}
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            archive_meta = json.load(f)

    dataset_metadata = {
        **archive_meta,
        "dataset": "scifact",
        "split": split,
        "corpus_count": len(corpus),
        "queries_count": len(queries),
        "qrels_queries_count": len(qrels),
        "qrels_pairs_count": sum(len(d) for d in qrels.values()),
    }

    return SciFactDataset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        metadata=dataset_metadata,
    )


def process_and_save_scifact(
    raw_dir: Path | str = "data/raw/scifact",
    processed_dir: Path | str = "data/processed/scifact",
    split: str = "test",
) -> Path:
    """Parse raw SciFact and save processed, documented artifacts.

    Saves:
        - corpus.jsonl (id, title, text, metadata)
        - queries.jsonl (id, text, metadata, evidence)
        - qrels_{split}.jsonl (query_id, doc_id, score)
        - metadata.json (dataset provenance, split, record counts, checksum)
    """
    raw_path = Path(raw_dir)
    out_path = Path(processed_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = load_scifact_dataset(raw_path, split=split)

    # 1. Save corpus
    corpus_out = out_path / "corpus.jsonl"
    with open(corpus_out, "w", encoding="utf-8") as f:
        for doc in dataset.corpus.values():
            record = {
                "_id": doc.id,
                "title": doc.title,
                "text": doc.text,
                "metadata": doc.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2. Save queries
    queries_out = out_path / "queries.jsonl"
    with open(queries_out, "w", encoding="utf-8") as f:
        for q in dataset.queries.values():
            record = {
                "_id": q.id,
                "text": q.text,
                "metadata": q.metadata,
                "evidence": [asdict(ev) for ev in q.evidence],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 3. Save qrels
    qrels_out = out_path / f"qrels_{split}.jsonl"
    with open(qrels_out, "w", encoding="utf-8") as f:
        for qid, doc_dict in dataset.qrels.items():
            for docid, score in doc_dict.items():
                record = {"query_id": qid, "doc_id": docid, "score": score}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 4. Save metadata
    meta_out = out_path / "metadata.json"
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(dataset.metadata, f, indent=2)

    logger.info(f"Successfully processed and saved SciFact to {out_path}.")
    return out_path
