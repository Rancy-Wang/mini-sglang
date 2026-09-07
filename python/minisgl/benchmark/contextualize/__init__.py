"""Manifest-driven correctness and serving benchmarks for Contextualize workloads."""

from .manifest import (
    MESSAGE_SHAPES,
    CaptureRecord,
    CaseMetadata,
    ManifestCase,
    MatchConfig,
    OracleResult,
    classify_assistant_message,
    coverage_matrix,
    dump_jsonl,
    load_capture_records,
    load_manifest,
)

__all__ = [
    "MESSAGE_SHAPES",
    "CaptureRecord",
    "CaseMetadata",
    "ManifestCase",
    "MatchConfig",
    "OracleResult",
    "classify_assistant_message",
    "coverage_matrix",
    "dump_jsonl",
    "load_capture_records",
    "load_manifest",
]
