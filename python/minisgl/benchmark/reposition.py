from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from minisgl.kernel.radix import (
    fast_compare_key,
    fast_compare_radix_records,
    fast_compare_retry_radix_records_plan,
    radix_record_compare_backend,
)
from minisgl.kernel.radix_reposition import (
    DELTA_KIND,
    REPOSITION_KIND,
    TOKEN_KIND,
    RadixRepositionInput,
    RadixRepositionLayout,
    compile_radix_reposition_layout,
    compile_radix_reposition_layout_batch,
)

DEFAULT_TOKEN_COUNTS = (8_192, 32_768, 65_536, 131_072, 262_144)
DEFAULT_CONCURRENCIES = (1, 4)


@dataclass(frozen=True)
class Timing:
    p50_ms: float
    p95_ms: float


@dataclass(frozen=True)
class BenchmarkResult:
    token_count: int
    concurrency: int
    event_count: int
    compile_aot: Timing
    compile_python: Timing
    compile_speedup: float
    exact_structured: Timing
    legacy_flat: Timing
    exact_throughput_ratio: float
    retry_plan: Timing
    retry_changed_pages: int
    host_to_device_bytes: int
    device_to_host_bytes: int
    compile_match_p95_ttft_percent: float | None


def make_benchmark_input(token_count: int, event_count: int = 8) -> RadixRepositionInput:
    """Build deterministic interleaved Drop/Reposition CPU tensors."""

    if token_count < 4:
        raise ValueError("token_count must be at least 4.")
    if event_count < 0 or event_count >= token_count // 2:
        raise ValueError("event_count must fit inside the token stream.")
    if event_count == 0:
        event_offsets: list[int] = []
    else:
        stride = token_count // (event_count + 1)
        event_offsets = [stride * (event + 1) for event in range(event_count)]

    flat_ranges = [value for event in range(event_count) for value in (event, event + 1)]
    return RadixRepositionInput(
        token_ids=torch.arange(token_count, dtype=torch.int64, device="cpu") % 200_000,
        drop_insert_offsets=torch.tensor(event_offsets, dtype=torch.int32, device="cpu"),
        drop_range_offsets=torch.arange(event_count + 1, dtype=torch.int32, device="cpu"),
        drop_ranges=torch.tensor(flat_ranges, dtype=torch.int32, device="cpu"),
        delta_marker_ids=-torch.arange(1, event_count + 1, dtype=torch.int32, device="cpu"),
        reposition_raw_boundaries=torch.tensor(
            [offset - 1 for offset in event_offsets], dtype=torch.int32, device="cpu"
        ),
        reposition_insert_offsets=torch.tensor(event_offsets, dtype=torch.int32, device="cpu"),
    )


def _python_reference_records(request: RadixRepositionInput) -> list[list[int]]:
    token_ids = request.token_ids.tolist()
    insertions = request.drop_insert_offsets.tolist()
    range_offsets = request.drop_range_offsets.tolist()
    ranges = request.drop_ranges.view(-1, 2).tolist()
    marker_ids = request.delta_marker_ids.tolist()
    reposition_boundaries = request.reposition_raw_boundaries.tolist()
    drop_by_offset = {
        insertion: (
            ranges[range_offsets[event] : range_offsets[event + 1]],
            marker_ids[event],
        )
        for event, insertion in enumerate(insertions)
    }
    reposition_by_offset = {
        boundary + 1: (event, boundary) for event, boundary in enumerate(reposition_boundaries)
    }
    active: list[int] = []
    positions = [-1] * len(token_ids)
    repos_info = [-1] * len(token_ids)
    effective = [False] * len(reposition_boundaries)
    current_reposition = -1
    next_position = 0

    for insertion in range(len(token_ids) + 1):
        drop = drop_by_offset.get(insertion)
        if drop is not None:
            dropped = {token for start, end in drop[0] for token in range(start, end)}
            active = [token for token in active if token not in dropped]

        reposition = reposition_by_offset.get(insertion)
        if reposition is not None:
            event, boundary = reposition
            if any(positions[token] != rank for rank, token in enumerate(active)):
                effective[event] = True
                for rank, token in enumerate(active):
                    if positions[token] != rank:
                        positions[token] = rank
                        repos_info[token] = boundary
                current_reposition = boundary
                next_position = len(active)

        if insertion < len(token_ids):
            positions[insertion] = next_position
            repos_info[insertion] = current_reposition
            active.append(insertion)
            next_position += 1

    records: list[list[int]] = []
    drop_event = 0
    reposition_event = 0
    for insertion, token_id in enumerate(token_ids + [None]):
        if drop_event < len(insertions) and insertions[drop_event] == insertion:
            records.append([DELTA_KIND, marker_ids[drop_event], -1, -1])
            drop_event += 1
        if (
            reposition_event < len(reposition_boundaries)
            and reposition_boundaries[reposition_event] + 1 == insertion
        ):
            if effective[reposition_event]:
                records.append([REPOSITION_KIND, reposition_boundaries[reposition_event], -1, -1])
            reposition_event += 1
        if token_id is not None:
            records.append([TOKEN_KIND, int(token_id), repos_info[insertion], positions[insertion]])
    return records


def _measure(function: Callable[[], object], repeats: int) -> Timing:
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        durations.append((time.perf_counter_ns() - start) / 1_000_000)
    ordered = sorted(durations)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return Timing(statistics.median(ordered), ordered[p95_index])


def _compile_one(request: RadixRepositionInput) -> RadixRepositionLayout:
    return compile_radix_reposition_layout(
        request.token_ids,
        request.drop_insert_offsets,
        request.drop_range_offsets,
        request.drop_ranges,
        request.delta_marker_ids,
        request.reposition_raw_boundaries,
        request.reposition_insert_offsets,
    )


def benchmark_case(
    token_count: int,
    concurrency: int,
    *,
    event_count: int = 8,
    repeats: int = 7,
    ttft_ms: float | None = None,
) -> BenchmarkResult:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive.")
    if ttft_ms is not None and ttft_ms <= 0:
        raise ValueError("ttft_ms must be positive when provided.")
    request = make_benchmark_input(token_count, event_count)
    requests = tuple(request for _ in range(concurrency))
    layout = _compile_one(request)
    reference = _python_reference_records(request)
    if layout.records.tolist() != reference:
        raise RuntimeError("AOT Radix layout disagrees with the independent Python oracle.")

    compile_aot = _measure(
        lambda: compile_radix_reposition_layout_batch(
            requests,
            max_workers=concurrency,
        ),
        repeats,
    )
    compile_python = _measure(
        lambda: tuple(_python_reference_records(item) for item in requests),
        repeats,
    )
    exact_target = layout.records.clone()
    exact_structured = _measure(
        lambda: fast_compare_radix_records(layout.records, exact_target),
        repeats,
    )
    legacy_flat = _measure(
        lambda: fast_compare_key(layout.records.view(-1), exact_target.view(-1)),
        repeats,
    )

    retry_target = layout.records.clone()
    token_rows = retry_target[:, 0] == TOKEN_KIND
    retry_target[token_rows, 2] += 1
    retry_target[token_rows, 3] += 1
    retry_plan_holder: list[torch.Tensor] = []

    def compile_retry_plan() -> None:
        _, plan = fast_compare_retry_radix_records_plan(
            layout.records,
            retry_target,
            layout.key_to_token,
            layout.key_to_token,
        )
        retry_plan_holder[:] = [plan]

    retry_plan = _measure(compile_retry_plan, repeats)
    compile_retry_plan()
    changed_pages = len(retry_plan_holder[0])
    exact_ratio = legacy_flat.p50_ms / max(exact_structured.p50_ms, 1e-9)
    speedup = compile_python.p50_ms / max(compile_aot.p50_ms, 1e-9)
    combined_p95 = compile_aot.p95_ms + exact_structured.p95_ms
    ttft_percent = None if ttft_ms is None else 100 * combined_p95 / ttft_ms
    return BenchmarkResult(
        token_count=token_count,
        concurrency=concurrency,
        event_count=event_count,
        compile_aot=compile_aot,
        compile_python=compile_python,
        compile_speedup=speedup,
        exact_structured=exact_structured,
        legacy_flat=legacy_flat,
        exact_throughput_ratio=exact_ratio,
        retry_plan=retry_plan,
        retry_changed_pages=changed_pages,
        host_to_device_bytes=0,
        device_to_host_bytes=0,
        compile_match_p95_ttft_percent=ttft_percent,
    )


def run_benchmark(
    token_counts: Sequence[int] = DEFAULT_TOKEN_COUNTS,
    concurrencies: Sequence[int] = DEFAULT_CONCURRENCIES,
    *,
    event_count: int = 8,
    repeats: int = 7,
    ttft_ms: float | None = None,
) -> dict[str, object]:
    results = [
        benchmark_case(
            token_count,
            concurrency,
            event_count=event_count,
            repeats=repeats,
            ttft_ms=ttft_ms,
        )
        for token_count in token_counts
        for concurrency in concurrencies
    ]
    return {
        "schema_version": 1,
        "compare_backend": radix_record_compare_backend(),
        "thresholds": {
            "compiler_speedup_min": 5.0,
            "exact_throughput_ratio_min": 0.60,
            "compile_match_p95_ttft_percent_max": 2.0,
            "no_reposition_ttft_overhead_percent_max": 3.0,
        },
        "measurement_notes": {
            "compile_match_p95_ttft_percent": (
                "available only when --ttft-ms supplies an end-to-end TTFT baseline"
            ),
            "no_reposition_ttft_overhead_percent": (
                "measured by the separate two-model end-to-end validation"
            ),
        },
        "results": [asdict(result) for result in results],
    }


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU Reposition Radix compilation.")
    parser.add_argument("--token-counts", default=",".join(map(str, DEFAULT_TOKEN_COUNTS)))
    parser.add_argument("--concurrencies", default=",".join(map(str, DEFAULT_CONCURRENCIES)))
    parser.add_argument("--event-count", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--ttft-ms", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_benchmark(
        _parse_ints(args.token_counts),
        _parse_ints(args.concurrencies),
        event_count=args.event_count,
        repeats=args.repeats,
        ttft_ms=args.ttft_ms,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
