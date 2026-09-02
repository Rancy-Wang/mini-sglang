from __future__ import annotations

import pytest

pytest.importorskip("tvm_ffi")

from minisgl.benchmark.reposition import make_benchmark_input, run_benchmark


def test_reposition_benchmark_emits_machine_readable_cpu_metrics() -> None:
    report = run_benchmark((128,), (1,), event_count=3, repeats=2, ttft_ms=100.0)

    assert report["schema_version"] == 1
    assert report["compare_backend"] in {"portable", "neon", "avx2", "avx512"}
    result = report["results"][0]
    assert result["token_count"] == 128
    assert result["concurrency"] == 1
    assert result["retry_changed_pages"] > 0
    assert result["host_to_device_bytes"] == 0
    assert result["device_to_host_bytes"] == 0
    assert result["compile_match_p95_ttft_percent"] is not None


def test_reposition_benchmark_rejects_impossible_event_density() -> None:
    with pytest.raises(ValueError, match="event_count"):
        make_benchmark_input(8, 4)
