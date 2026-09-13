from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).parents[2] / "scripts" / "verify_occurrence_usage.py"
_spec = importlib.util.spec_from_file_location("usage_oracle", _path)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
audit_request = _module.audit_request


def _trace():
    return {
        "uid": 1,
        "initial_pages": [
            {"page": 10, "raw": 0, "position": 0},
            {"page": 11, "raw": 1, "position": 1},
            {"page": 12, "raw": 2, "position": 2},
        ],
        "visible_until": [3, 5, 5, 5],
        "chunks": [{
            "birth_writes": [{"page": 13, "raw": 3, "position": 2}],
            "transforms": [
                {"source": 10, "destination": 20, "old": 0, "new": 7},
                {"source": 11, "destination": 21, "old": 1, "new": 1},
                {"source": 12, "destination": 22, "old": 2, "new": 1},
            ],
            "attention_reads": [{"query_raw": 3, "pages": [21, 22, 13]}],
        }],
    }


def test_usage_excludes_cacheback_only_rotation_and_same_position_copy():
    report = audit_request(_trace())
    assert report["expected"] == dict(cached_tokens=1, repos_tokens=1, drop_skipped_tokens=1)
    assert report["rotated_raw"] == [2]


def test_usage_deduplicates_repeated_attention_reads():
    trace = _trace()
    trace["chunks"][0]["attention_reads"] *= 3
    assert audit_request(trace)["expected"]["repos_tokens"] == 1


def test_usage_ignores_rotation_of_visible_token_when_only_unrotated_page_is_read():
    trace = _trace()
    trace["chunks"][0]["attention_reads"][0]["pages"] = [21, 12, 13]
    assert audit_request(trace)["expected"] == dict(cached_tokens=2, repos_tokens=0, drop_skipped_tokens=1)


def test_usage_rejects_bad_visibility():
    trace = _trace()
    trace["chunks"][0]["attention_reads"][0]["pages"].append(20)
    with pytest.raises(ValueError, match="Invisible"):
        audit_request(trace)


def test_usage_rejects_wrong_source_position():
    trace = _trace()
    trace["chunks"][0]["transforms"][0]["old"] = 99
    with pytest.raises(ValueError, match="source position"):
        audit_request(trace)


def test_segment_trace_is_equivalent_to_expanded_causal_reads():
    trace = _trace()
    chunk = trace["chunks"][0]
    chunk["birth_writes"].append({"page": 14, "raw": 4, "position": 3})
    trace["visible_until"].append(5)
    chunk["attention_reads"].append({"query_raw": 4, "pages": [21, 22, 13, 14]})
    expanded = audit_request(trace)
    chunk["attention_reads"] = []
    chunk["attention_segments"] = [
        {"query_start": 3, "query_end": 5, "pages": [21, 22, 13, 14]}
    ]
    assert audit_request(trace) == expanded
    trace["visible_until"][1] = 4
    with pytest.raises(ValueError, match="visibility boundary"):
        audit_request(trace)


@pytest.mark.parametrize("use_rotated", [False, True])
def test_production_usage_agrees_with_page_lineage_oracle(use_rotated):
    import torch
    from minisgl.core import Req
    from minisgl.attention.base import build_occurrence_attention_batch

    req = object.__new__(Req)
    t = lambda values: torch.tensor(values, dtype=torch.int32)
    req.reposition_execution_mode = "paged-occurrence"
    req.occurrence_raw_tokens = t([0, 1, 2, 3, 0, 1, 2])
    req.occurrence_positions = t([0, 1, 2, 2, 7, 1, 1])
    req.occurrence_pages = t([10, 11, 12, 13, 20, 21, 22])
    req.occurrence_segment_query_starts = t([3])
    req.occurrence_segment_query_ends = t([4])
    req.occurrence_segment_key_offsets = t([0, 3])
    req.occurrence_segment_key_indices = t([5, 6 if use_rotated else 2, 3])
    req.occurrence_initial_source_positions = t([0, 1, 2])
    req.occurrence_repositioned_cached_mask = torch.zeros(3, dtype=torch.bool)
    req.cached_len = req.initial_active_cached_len = req.radix_cached_tokens = 3
    req.device_len = 4
    batch = build_occurrence_attention_batch([req])
    req.record_context_cache_usage(batch.cached_tokens[0], batch.cached_positions[0])
    trace = _trace()
    if not use_rotated:
        trace["chunks"][0]["attention_reads"][0]["pages"] = [21, 12, 13]
    expected = audit_request(trace)["expected"]
    assert dict(cached_tokens=req.reported_cached_tokens,
                repos_tokens=req.reported_repos_tokens,
                drop_skipped_tokens=req.drop_skipped_tokens) == expected
