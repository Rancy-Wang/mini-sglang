from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from minisgl.core import SamplingParams
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
from minisgl.message import TokenizeMsg, WarmupAckMsg
from minisgl.tokenizer.reposition_sequence import RepositionSequenceState
from minisgl.tokenizer.tokenize import TokenizedResult

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
    compile_python: Timing | None
    compile_speedup: float | None
    exact_structured: Timing
    legacy_flat: Timing
    exact_throughput_ratio: float
    retry_plan: Timing
    retry_changed_pages: int
    logical_reposition_stage_count: int
    scheduler_dispatch_count: int | None
    timeline_transition_count: int
    timeline_metadata_bytes: int
    dense_stage_snapshot_bytes: int
    reposition_ipc_tensor_bytes: int | None
    measured_radix_key_h2d_bytes: int
    measured_radix_key_d2h_bytes: int
    retry_metadata_payload_bytes: int
    measured_retry_metadata_h2d_bytes: int
    measured_retry_metadata_d2h_bytes: int
    measured_ordinary_cache_h2d_bytes: int
    measured_ordinary_cache_d2h_bytes: int
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
    reposition_boundaries = request.reposition_raw_boundaries.tolist()
    drop_by_offset = {
        insertion: ranges[range_offsets[event] : range_offsets[event + 1]]
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
            dropped = {token for start, end in drop for token in range(start, end)}
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
            for start, end in ranges[
                range_offsets[drop_event] : range_offsets[drop_event + 1]
            ]:
                records.append([DELTA_KIND, -start - 1, -end - 1, -1])
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
        request.reposition_raw_boundaries,
        request.reposition_insert_offsets,
    )


def _visible_until(request: RadixRepositionInput) -> torch.Tensor:
    result = torch.full(
        (len(request.token_ids),),
        torch.iinfo(torch.int32).max,
        dtype=torch.int32,
        device="cpu",
    )
    ranges = request.drop_ranges.view(-1, 2)
    for event, insertion in enumerate(request.drop_insert_offsets.tolist()):
        begin = int(request.drop_range_offsets[event])
        end = int(request.drop_range_offsets[event + 1])
        for start, finish in ranges[begin:end].tolist():
            result[int(start) : int(finish)] = int(insertion)
    return result


def _measure_scheduler_dispatch(
    request: RadixRepositionInput,
    layout: RadixRepositionLayout,
) -> tuple[int, int]:
    if len(layout.transition_offsets) <= 1:
        # The production path correctly bypasses RepositionSequenceState when
        # there is no effective Reposition.  It has one ordinary dispatch and
        # no Reposition IPC payload.
        return 1, 0
    token_ids = request.token_ids.to(torch.int32)
    tokenized = TokenizedResult(
        input_ids=token_ids[layout.keep_mask],
        true_positions=layout.positions[layout.keep_mask],
        raw_positions=torch.nonzero(layout.keep_mask, as_tuple=False).view(-1).to(torch.int32),
        radix_input_ids=layout.records[layout.token_to_key[layout.keep_mask]],
        radix_match_ids=layout.records,
        prefix_keep_mask=layout.keep_mask[:-1].to(torch.int32),
        prompt_tokens=len(token_ids),
        full_input_ids=token_ids,
        full_token_visible_until=_visible_until(request),
        full_keep_mask=layout.keep_mask.to(torch.int32),
        reposition_raw_boundaries=request.reposition_raw_boundaries,
        reposition_insert_offsets=request.reposition_insert_offsets,
        reposition_input_ids=token_ids,
        reposition_layout=layout,
        tokenize_invocations=1,
        chat_template_invocations=1,
    )
    state = RepositionSequenceState.pending(
        TokenizeMsg(
            uid=1,
            text="benchmark",
            sampling_params=SamplingParams(max_tokens=1),
            reposition=request.reposition_raw_boundaries.tolist(),
        ),
        tokenized,
    )
    state.activate(step_token_budget=len(token_ids))
    while True:
        message = state.build_next_msg()
        if not message.is_warmup:
            return message.context_stage_count, message.reposition_ipc_tensor_bytes
        state.accept_ack(
            WarmupAckMsg(
                uid=message.uid,
                hit_ratio=0.0,
                cached_tokens=0,
                drop_skipped_tokens=0,
                finished=True,
            )
        )


def benchmark_case(
    token_count: int,
    concurrency: int,
    *,
    event_count: int = 8,
    repeats: int = 7,
    ttft_ms: float | None = None,
    include_python_oracle: bool = True,
    include_sequence_metrics: bool = True,
) -> BenchmarkResult:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive.")
    if ttft_ms is not None and ttft_ms <= 0:
        raise ValueError("ttft_ms must be positive when provided.")
    request = make_benchmark_input(token_count, event_count)
    requests = tuple(request for _ in range(concurrency))
    layout = _compile_one(request)
    if include_python_oracle:
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
    compile_python = (
        _measure(
            lambda: tuple(_python_reference_records(item) for item in requests),
            repeats,
        )
        if include_python_oracle
        else None
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
    logical_stage_count = len(layout.transition_offsets)
    if include_sequence_metrics:
        scheduler_dispatch_count, ipc_tensor_bytes = _measure_scheduler_dispatch(request, layout)
    else:
        scheduler_dispatch_count = None
        ipc_tensor_bytes = None
    transition_count = len(layout.transition_raw_tokens)
    timeline_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            layout.birth_positions,
            layout.birth_stages,
            layout.transition_offsets,
            layout.transition_raw_tokens,
            layout.transition_old_positions,
            layout.transition_new_positions,
            layout.effective_reposition_stages,
        )
    )
    int32_bytes = torch.empty((), dtype=torch.int32).element_size()
    dense_snapshot_bytes = token_count * logical_stage_count * int32_bytes
    exact_ratio = legacy_flat.p50_ms / max(exact_structured.p50_ms, 1e-9)
    speedup = (
        compile_python.p50_ms / max(compile_aot.p50_ms, 1e-9)
        if compile_python is not None
        else None
    )
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
        logical_reposition_stage_count=logical_stage_count,
        scheduler_dispatch_count=scheduler_dispatch_count,
        timeline_transition_count=transition_count,
        timeline_metadata_bytes=timeline_bytes,
        dense_stage_snapshot_bytes=dense_snapshot_bytes,
        reposition_ipc_tensor_bytes=ipc_tensor_bytes,
        measured_radix_key_h2d_bytes=0,
        measured_radix_key_d2h_bytes=0,
        retry_metadata_payload_bytes=changed_pages * 5 * int32_bytes,
        measured_retry_metadata_h2d_bytes=0,
        measured_retry_metadata_d2h_bytes=0,
        measured_ordinary_cache_h2d_bytes=0,
        measured_ordinary_cache_d2h_bytes=0,
        compile_match_p95_ttft_percent=ttft_percent,
    )


def run_benchmark(
    token_counts: Sequence[int] = DEFAULT_TOKEN_COUNTS,
    concurrencies: Sequence[int] = DEFAULT_CONCURRENCIES,
    *,
    event_count: int = 8,
    repeats: int = 7,
    ttft_ms: float | None = None,
    include_python_oracle: bool = True,
    include_sequence_metrics: bool = True,
) -> dict[str, object]:
    results = [
        benchmark_case(
            token_count,
            concurrency,
            event_count=event_count,
            repeats=repeats,
            ttft_ms=ttft_ms,
            include_python_oracle=include_python_oracle,
            include_sequence_metrics=include_sequence_metrics,
        )
        for token_count in token_counts
        for concurrency in concurrencies
    ]
    thresholds = {
        "compiler_speedup_min": 5.0,
        "exact_throughput_ratio_min": 0.60,
        "compile_match_p95_ttft_percent_max": 2.0,
        "no_reposition_ttft_overhead_percent_max": 3.0,
    }
    threshold_failures: list[dict[str, object]] = []
    for result in results:
        failures: list[str] = []
        if (
            result.compile_speedup is not None
            and result.compile_speedup < thresholds["compiler_speedup_min"]
        ):
            failures.append("compiler_speedup")
        if result.exact_throughput_ratio < thresholds["exact_throughput_ratio_min"]:
            failures.append("exact_throughput_ratio")
        if (
            result.compile_match_p95_ttft_percent is not None
            and result.compile_match_p95_ttft_percent
            > thresholds["compile_match_p95_ttft_percent_max"]
        ):
            failures.append("compile_match_p95_ttft_percent")
        if failures:
            threshold_failures.append(
                {
                    "token_count": result.token_count,
                    "concurrency": result.concurrency,
                    "failed": failures,
                }
            )
    return {
        "schema_version": 3,
        "compare_backend": radix_record_compare_backend(),
        "thresholds": thresholds,
        "threshold_evaluation": {
            "passed": not threshold_failures,
            "failures": threshold_failures,
        },
        "measurement_notes": {
            "compile_match_p95_ttft_percent": (
                "available only when --ttft-ms supplies an end-to-end TTFT baseline"
            ),
            "no_reposition_ttft_overhead_percent": (
                "measured by the separate two-model end-to-end validation"
            ),
            "transfer_bytes": (
                "This CPU benchmark performs no device transfer. measured_* fields are zero; "
                "retry_metadata_payload_bytes is the exact five-int32 payload size that the "
                "scheduler would transfer for the compiled Retry plan."
            ),
            "reposition_ipc_tensor_bytes": (
                "exact cumulative tensor buffer bytes serialized by Tokenizer-to-Scheduler "
                "Reposition dispatches; scalar protocol framing is excluded; omitted in "
                "--production-only mode so performance timing does not materialize an "
                "additional sequence of cumulative tensor snapshots"
            ),
            "python_oracle": (
                "omitted in --production-only mode; correctness remains a separate focused gate"
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
    parser.add_argument(
        "--production-only",
        action="store_true",
        help="Skip the intentionally slow Python oracle for large production-scale timings.",
    )
    parser.add_argument(
        "--enforce-thresholds",
        action="store_true",
        help="Exit non-zero when any measured CPU threshold fails.",
    )
    args = parser.parse_args(argv)
    report = run_benchmark(
        _parse_ints(args.token_counts),
        _parse_ints(args.concurrencies),
        event_count=args.event_count,
        repeats=args.repeats,
        ttft_ms=args.ttft_ms,
        include_python_oracle=not args.production_only,
        include_sequence_metrics=not args.production_only,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.enforce_thresholds and not report["threshold_evaluation"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
