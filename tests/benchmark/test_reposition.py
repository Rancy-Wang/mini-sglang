from __future__ import annotations

import pytest

pytest.importorskip("tvm_ffi")

from minisgl.benchmark.reposition import make_benchmark_input, run_benchmark


def test_reposition_benchmark_emits_machine_readable_cpu_metrics() -> None:
    report = run_benchmark((128,), (1,), event_count=3, repeats=2, ttft_ms=100.0)

    assert report["schema_version"] == 3
    assert report["compare_backend"] in {"portable", "neon", "avx2", "avx512"}
    result = report["results"][0]
    assert result["token_count"] == 128
    assert result["concurrency"] == 1
    assert result["retry_changed_pages"] > 0
    assert result["logical_reposition_stage_count"] == 4
    assert result["scheduler_dispatch_count"] == 4
    assert result["timeline_transition_count"] > 0
    assert result["timeline_metadata_bytes"] > 0
    assert result["dense_stage_snapshot_bytes"] == 128 * 4 * 4
    assert result["reposition_ipc_tensor_bytes"] > 0
    assert result["measured_radix_key_h2d_bytes"] == 0
    assert result["measured_radix_key_d2h_bytes"] == 0
    assert result["retry_metadata_payload_bytes"] == result["retry_changed_pages"] * 5 * 4
    assert result["measured_retry_metadata_h2d_bytes"] == 0
    assert result["measured_retry_metadata_d2h_bytes"] == 0
    assert result["measured_ordinary_cache_h2d_bytes"] == 0
    assert result["measured_ordinary_cache_d2h_bytes"] == 0
    assert result["compile_match_p95_ttft_percent"] is not None
    assert isinstance(report["threshold_evaluation"]["passed"], bool)


def test_reposition_benchmark_rejects_impossible_event_density() -> None:
    with pytest.raises(ValueError, match="event_count"):
        make_benchmark_input(8, 4)


def test_production_only_benchmark_does_not_claim_python_or_transfer_timings() -> None:
    report = run_benchmark(
        (128,),
        (1,),
        event_count=3,
        repeats=1,
        include_python_oracle=False,
        include_sequence_metrics=False,
    )

    result = report["results"][0]
    assert result["compile_python"] is None
    assert result["compile_speedup"] is None
    assert result["scheduler_dispatch_count"] is None
    assert result["reposition_ipc_tensor_bytes"] is None
    assert result["measured_retry_metadata_h2d_bytes"] == 0
