"""Dataset loaders and preprocessing modules."""

from .scifact import (
    SciFactDataset,
    SciFactDocument,
    SciFactQuery,
    download_scifact,
    load_scifact_dataset,
    process_and_save_scifact,
)
from .evidence_mapper import (
    EvidenceMappingRecord,
    build_evidence_mappings,
    save_evidence_mappings,
    load_verified_evidence_ground_truth,
    match_sentence_in_document,
    normalize_for_matching,
)

__all__ = [
    "SciFactDataset",
    "SciFactDocument",
    "SciFactQuery",
    "download_scifact",
    "load_scifact_dataset",
    "process_and_save_scifact",
    "EvidenceMappingRecord",
    "build_evidence_mappings",
    "save_evidence_mappings",
    "load_verified_evidence_ground_truth",
    "match_sentence_in_document",
    "normalize_for_matching",
]
