from __future__ import annotations

import argparse
import asyncio
import copy
import gzip
import json
import shutil
import statistics
import subprocess
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import httpx
from minisgl.benchmark.reposition_bcp import (
    CapturedRequest,
    ReplayTask,
    append_gzip_jsonl,
    apply_rolling_interface,
    attribute_issues,
    audit_parsed_response,
    discover_captured_requests,
    load_rollout_prompt_token_hints,
    load_trajectory_replay_tasks,
    materialize_fixed_request_cases,
    render_text_trajectory,
    request_sha256,
    select_replay_tasks,
    select_task_set,
)
from minisgl.benchmark.reposition_trajectory import (
    METRIC_FIELDS,
    inspect_metrics,
    parse_response_bytes,
)

from scripts.profile_reposition_matrix import (
    create_profile_bootstrap,
    detect_slowdown,
    summarize_profile,
)
from scripts.run_request_matrix import load_server_configs, running_server

_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


@dataclass(frozen=True)
class ExperimentCell:
    name: str
    mode: str
    concurrency: int
    repetitions: int
    endpoints: tuple[str, ...]
    server_config: Path | None
    request_overrides: dict[str, Any]
    baseline_cell: str | None
    framework: str | None
    profile_on_slowdown: bool
    nsys: bool
    telemetry: bool
    request_selection: str
    warmup: bool


def _write_gzip_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as output:
        output.write(data)


def _write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    _write_gzip_bytes(path, data)


def _chat_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


def _expected_assistant(task: ReplayTask, index: int) -> dict[str, Any] | None:
    current = task.requests[index].request.get("messages")
    if not isinstance(current, list):
        return None
    for successor in task.requests[index + 1 :]:
        messages = successor.request.get("messages")
        if not isinstance(messages, list) or messages[: len(current)] != current:
            continue
        for message in messages[len(current) :]:
            if isinstance(message, dict) and message.get("role") == "assistant":
                return message
    return None


async def _post_capture(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    request: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    started_ns = time.time_ns()
    first_byte_ns: int | None = None
    raw = bytearray()
    status_code: int | None = None
    response_headers: list[tuple[str, str]] = []
    error: dict[str, str] | None = None
    try:
        async with client.stream("POST", _chat_url(endpoint), json=request) as response:
            status_code = response.status_code
            response_headers = list(response.headers.multi_items())
            async for chunk in response.aiter_raw():
                if chunk and first_byte_ns is None:
                    first_byte_ns = time.time_ns()
                raw.extend(chunk)
        finished_ns = time.time_ns()
        if status_code != 200:
            decoded = bytes(raw).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {status_code}: {decoded[:1000]}")
        parsed = parse_response_bytes(bytes(raw), stream=request.get("stream") is True)
        return bytes(raw), {
            "ok": True,
            "status_code": status_code,
            "headers": response_headers,
            "client_started_ns": started_ns,
            "client_first_byte_ns": first_byte_ns,
            "client_finished_ns": finished_ns,
            **parsed,
        }
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    return bytes(raw), {
        "ok": False,
        "status_code": status_code,
        "headers": response_headers,
        "client_started_ns": started_ns,
        "client_first_byte_ns": first_byte_ns,
        "client_finished_ns": time.time_ns(),
        "error": error,
        "issues": ["transport:request_or_parse_failure"],
    }


async def replay_tasks(
    tasks: Sequence[ReplayTask],
    *,
    endpoints: Sequence[str],
    mode: str,
    concurrency: int,
    repetitions: int,
    output_dir: Path,
    request_overrides: dict[str, Any],
    request_timeout: float,
    request_selection: str = "all",
    warmup: bool = False,
    repetition_start: int = 1,
    require_server_metrics: bool = False,
) -> list[dict[str, Any]]:
    if not tasks or not endpoints:
        raise ValueError("tasks and endpoints must be non-empty")
    if concurrency < 1 or repetitions < 1 or repetition_start < 1:
        raise ValueError("concurrency and repetitions must be positive")
    if request_selection not in {"all", "last"}:
        raise ValueError("request_selection must be all or last")
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphores = [asyncio.Semaphore(concurrency) for _ in endpoints]
    records: list[dict[str, Any]] = []
    records_lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=request_timeout, trust_env=False) as client:
        if warmup:
            warmups = []
            for endpoint_index, endpoint in enumerate(endpoints):
                request, _ = apply_rolling_interface(tasks[0].requests[-1].request, mode=mode)
                request.update(copy.deepcopy(request_overrides))
                messages = request.get("messages", [])
                first_text = next(
                    (
                        message
                        for message in messages
                        if isinstance(message, dict) and isinstance(message.get("content"), str)
                    ),
                    None,
                )
                if first_text is not None:
                    first_text["content"] += (
                        "\n[R10 isolated warmup; do not reuse as benchmark input]"
                    )
                raw, response = await _post_capture(client, endpoint=endpoint, request=request)
                stem = f"endpoint-{endpoint_index:02d}"
                _write_gzip_json(output_dir / "warmup" / f"{stem}.request.json.gz", request)
                _write_gzip_bytes(output_dir / "warmup" / f"{stem}.response.bin.gz", raw)
                warmups.append({"endpoint": endpoint, "response": response})
            append_gzip_jsonl(output_dir / "warmup" / "result.jsonl.gz", warmups)

        async def run_task(endpoint_index: int, repetition: int, task: ReplayTask) -> None:
            async with semaphores[endpoint_index]:
                indexed_requests = list(enumerate(task.requests))
                if request_selection == "last":
                    indexed_requests = indexed_requests[-1:]
                for turn_index, capture in indexed_requests:
                    request, rolling_plan = apply_rolling_interface(capture.request, mode=mode)
                    request.update(copy.deepcopy(request_overrides))
                    expected = _expected_assistant(task, turn_index)
                    stem = (
                        f"endpoint-{endpoint_index:02d}/rep-{repetition:02d}/case-{task.case_id}/"
                        f"turn-{turn_index + 1:03d}"
                    )
                    _write_gzip_json(output_dir / "raw" / f"{stem}.request.json.gz", request)
                    raw, response = await _post_capture(
                        client, endpoint=endpoints[endpoint_index], request=request
                    )
                    _write_gzip_bytes(output_dir / "raw" / f"{stem}.response.bin.gz", raw)
                    issues = list(response.pop("issues", []))
                    audit: dict[str, Any] = {"issues": issues, "inspection": {}}
                    if response.get("ok"):
                        audit = audit_parsed_response(
                            response,
                            request=request,
                            expected=expected,
                        )
                        if require_server_metrics:
                            audit["issues"] = sorted(
                                set(audit["issues"])
                                | set(
                                    inspect_metrics(
                                        response.get("server_metrics"),
                                        stress=bool(request.get("reposition")),
                                        cold=bool(request.get("reposition")),
                                    )
                                )
                            )
                    record = {
                        "case_id": task.case_id,
                        "mode": mode,
                        "endpoint_index": endpoint_index,
                        "endpoint": endpoints[endpoint_index],
                        "repetition": repetition,
                        "turn": turn_index + 1,
                        "source_path": str(capture.path),
                        "source_provenance": capture.provenance,
                        "request_sha256": request_sha256(request),
                        "rolling_plan": rolling_plan.to_dict(),
                        "request": request,
                        "response": response,
                        "audit": audit,
                    }
                    async with records_lock:
                        records.append(record)

        await asyncio.gather(
            *(
                run_task(endpoint_index, repetition, task)
                for endpoint_index in range(len(endpoints))
                for repetition in range(repetition_start, repetition_start + repetitions)
                for task in tasks
            )
        )

    records.sort(
        key=lambda item: (
            item["endpoint_index"],
            item["repetition"],
            item["case_id"],
            item["turn"],
        )
    )
    append_gzip_jsonl(output_dir / "result.jsonl.gz", records)
    (output_dir / "trajectory.txt").write_text(render_text_trajectory(records), encoding="utf-8")
    return records


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    latencies_ms: list[float] = []
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    issue_counts: dict[str, int] = {}
    generated_tokens = 0
    first_start: int | None = None
    last_finish: int | None = None
    for record in records:
        response = record.get("response", {})
        started = response.get("client_started_ns")
        first = response.get("client_first_byte_ns")
        finished = response.get("client_finished_ns")
        if isinstance(started, int) and isinstance(finished, int):
            latencies_ms.append((finished - started) / 1_000_000)
            first_start = started if first_start is None else min(first_start, started)
            last_finish = finished if last_finish is None else max(last_finish, finished)
        metrics = response.get("server_metrics")
        metric_received = metrics.get("request_received_ns") if isinstance(metrics, dict) else None
        metric_first = (
            metrics.get("first_token_generated_ns") if isinstance(metrics, dict) else None
        )
        if isinstance(metric_received, int) and isinstance(metric_first, int):
            current_ttft = (metric_first - metric_received) / 1_000_000
        elif isinstance(started, int) and isinstance(first, int):
            current_ttft = (first - started) / 1_000_000
        else:
            current_ttft = None
        if current_ttft is not None:
            ttft_ms.append(current_ttft)
        usage = response.get("usage")
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if not isinstance(completion_tokens, int) and isinstance(metrics, dict):
            completion_tokens = metrics.get("completion_tokens")
        if isinstance(completion_tokens, int):
            generated_tokens += completion_tokens
            if (
                completion_tokens > 1
                and current_ttft is not None
                and isinstance(started, int)
                and isinstance(finished, int)
            ):
                decode_ms = (finished - started) / 1_000_000 - current_ttft
                tpot_ms.append(max(0.0, decode_ms) / (completion_tokens - 1))
        for issue in record.get("audit", {}).get("issues", []):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    def percentile(values: Sequence[float], quantile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = round((len(ordered) - 1) * quantile)
        return ordered[index]

    wall_seconds = (
        (last_finish - first_start) / 1_000_000_000
        if first_start is not None and last_finish is not None and last_finish > first_start
        else None
    )
    grouped: dict[tuple[int, int], list[float]] = {}
    for record in records:
        response = record.get("response", {})
        started = response.get("client_started_ns")
        finished = response.get("client_finished_ns")
        if isinstance(started, int) and isinstance(finished, int):
            key = (int(record.get("endpoint_index", 0)), int(record.get("repetition", 1)))
            grouped.setdefault(key, []).append((finished - started) / 1_000_000)
    repetition_samples = [
        {
            "endpoint_index": endpoint,
            "repetition": repetition,
            "e2e_p50_ms": statistics.median(values),
        }
        for (endpoint, repetition), values in sorted(grouped.items())
    ]
    return {
        "requests": len(records),
        "successful": sum(bool(record.get("response", {}).get("ok")) for record in records),
        "issues": issue_counts,
        "latency_ms": {
            "p50": percentile(latencies_ms, 0.5),
            "p95": percentile(latencies_ms, 0.95),
        },
        "ttft_ms": {"p50": percentile(ttft_ms, 0.5), "p95": percentile(ttft_ms, 0.95)},
        "tpot_ms": {"p50": percentile(tpot_ms, 0.5), "p95": percentile(tpot_ms, 0.95)},
        "generated_tokens": generated_tokens,
        "wall_seconds": wall_seconds,
        "request_throughput_per_second": len(records) / wall_seconds if wall_seconds else None,
        "output_token_throughput_per_second": (
            generated_tokens / wall_seconds if wall_seconds else None
        ),
        "repetition_samples": repetition_samples,
    }


def _load_manifest(path: Path) -> tuple[list[Path], list[str], int, list[ExperimentCell]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment manifest must be an object")
    base = path.parent
    roots = []
    for value in payload.get("capture_roots", []):
        root = Path(value).expanduser()
        roots.append((base / root).resolve() if not root.is_absolute() else root)
    preferred = [str(item) for item in payload.get("preferred_case_ids", [])]
    task_limit = int(payload.get("task_limit", 20))
    cells: list[ExperimentCell] = []
    for index, item in enumerate(payload.get("cells", [])):
        if not isinstance(item, dict):
            raise ValueError(f"cells[{index}] must be an object")
        config_value = item.get("server_config")
        server_config = None
        if config_value is not None:
            candidate = Path(config_value).expanduser()
            server_config = (
                (base / candidate).resolve() if not candidate.is_absolute() else candidate
            )
        endpoints = tuple(str(value) for value in item.get("endpoints", []))
        if bool(server_config) == bool(endpoints):
            raise ValueError(f"cells[{index}] must set exactly one of endpoints or server_config")
        cells.append(
            ExperimentCell(
                name=str(item["name"]),
                mode=str(item["mode"]),
                concurrency=int(item["concurrency"]),
                repetitions=int(item.get("repetitions", 3)),
                endpoints=endpoints,
                server_config=server_config,
                request_overrides=copy.deepcopy(item.get("request_overrides", {})),
                baseline_cell=(
                    str(item["baseline_cell"]) if item.get("baseline_cell") is not None else None
                ),
                framework=(str(item["framework"]) if item.get("framework") is not None else None),
                profile_on_slowdown=bool(item.get("profile_on_slowdown", False)),
                nsys=bool(item.get("nsys", True)),
                telemetry=bool(item.get("telemetry", False)),
                request_selection=str(item.get("request_selection", "all")),
                warmup=bool(item.get("warmup", False)),
            )
        )
        if cells[-1].request_selection not in {"all", "last"}:
            raise ValueError(f"cells[{index}].request_selection must be all or last")
    if not roots or not cells:
        raise ValueError("manifest needs capture_roots and cells")
    return roots, preferred, task_limit, cells


def _prepare_tasks(
    roots: Sequence[Path], preferred: Sequence[str], task_limit: int
) -> list[ReplayTask]:
    captures: list[CapturedRequest] = []
    reconstructed: list[ReplayTask] = []
    all_hints: dict[str, int] = {}
    for root in roots:
        captures.extend(discover_captured_requests(root))
        trajectories = root / "trajectories.jsonl"
        rollouts = root / "rollouts.jsonl"
        if trajectories.is_file():
            hints = load_rollout_prompt_token_hints([rollouts]) if rollouts.is_file() else {}
            all_hints.update(hints)
            reconstructed.extend(
                load_trajectory_replay_tasks([trajectories], prompt_token_hints=hints)
            )
    raw_tasks = select_replay_tasks(
        captures,
        limit=task_limit,
        preferred_case_ids=preferred,
        minimum_requests=6,
    )
    raw_tasks = [
        ReplayTask(task.case_id, task.requests, all_hints.get(task.case_id)) for task in raw_tasks
    ]
    by_case = {task.case_id: task for task in reconstructed}
    by_case.update({task.case_id: task for task in raw_tasks})
    tasks = select_task_set(
        by_case.values(),
        limit=task_limit,
        preferred_case_ids=preferred,
        minimum_requests=6,
    )
    if len(tasks) != task_limit:
        raise ValueError(f"requested {task_limit} eligible BCP tasks, found {len(tasks)}")
    return tasks


def _cell_repetition_medians(records: Sequence[dict[str, Any]]) -> list[float]:
    by_repetition: dict[int, list[float]] = {}
    for record in records:
        response = record.get("response", {})
        started = response.get("client_started_ns")
        finished = response.get("client_finished_ns")
        if not isinstance(started, int) or not isinstance(finished, int):
            continue
        repetition = int(record.get("repetition", 1))
        by_repetition.setdefault(repetition, []).append((finished - started) / 1_000_000)
    if len(by_repetition) < 3:
        raise ValueError("slowdown comparison requires three completed repetitions")
    return [statistics.median(values) for _, values in sorted(by_repetition.items())]


def _run_profile_cell(
    cell: ExperimentCell,
    *,
    tasks: Sequence[ReplayTask],
    output_dir: Path,
    startup_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
) -> dict[str, Any]:
    if cell.server_config is None or cell.framework not in {"minisgl", "sglang"}:
        return {
            "status": "blocked",
            "reason": "automatic profile requires server_config and framework",
        }
    profile_env = create_profile_bootstrap(output_dir, framework=cell.framework, nvtx=True)
    configs = []
    nsys_path = shutil.which("nsys") if cell.nsys else None
    for config in load_server_configs(cell.server_config):
        environment = {**config.env, **profile_env}
        argv = config.argv
        if nsys_path is not None:
            trace = output_dir / f"nsys-{config.config_id}"
            argv = (
                nsys_path,
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--sample=none",
                "--force-overwrite=true",
                "--output",
                str(trace),
                *argv,
            )
        configs.append(replace(config, argv=argv, env=environment))

    log_dir = output_dir / "server-logs"
    log_dir.mkdir()
    endpoints: list[str] = []
    with ExitStack() as stack:
        for config in configs:
            stack.enter_context(
                running_server(
                    config,
                    log_dir / f"{config.config_id}.log",
                    startup_timeout,
                    shutdown_timeout,
                )
            )
            endpoints.append(config.endpoint_url)
        records = asyncio.run(
            replay_tasks(
                tasks[:1],
                endpoints=endpoints,
                mode=cell.mode,
                concurrency=1,
                repetitions=1,
                output_dir=output_dir / "replay",
                request_overrides=cell.request_overrides,
                request_timeout=request_timeout,
                request_selection=cell.request_selection,
                warmup=cell.warmup,
            )
        )
    profile_paths = sorted(output_dir.glob("profile-*.jsonl"))
    report = {
        "status": "complete",
        "framework": cell.framework,
        "nsys_available": nsys_path is not None,
        "nsys_reports": [str(path) for path in sorted(output_dir.glob("nsys-*.nsys-rep"))],
        "replay": summarize_records(records),
        "function_profile": summarize_profile(profile_paths),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


@contextmanager
def _capture_telemetry(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = {
        "nvidia_smi_dmon": ["nvidia-smi", "dmon", "-s", "pucvmet", "-d", "1"],
        "pidstat": ["pidstat", "-rud", "-h", "1"],
    }
    processes: list[tuple[subprocess.Popen[bytes], Any]] = []
    availability: dict[str, bool] = {}
    try:
        for name, command in commands.items():
            executable = shutil.which(command[0])
            availability[name] = executable is not None
            if executable is None:
                continue
            log = (output_dir / f"{name}.log").open("wb")
            process = subprocess.Popen(
                [executable, *command[1:]],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((process, log))
        yield
    finally:
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, log in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
        (output_dir / "availability.json").write_text(
            json.dumps(availability, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _run_cell_replay(
    cell: ExperimentCell,
    *,
    tasks: Sequence[ReplayTask],
    output_dir: Path,
    startup_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if cell.telemetry:
            stack.enter_context(_capture_telemetry(output_dir / "telemetry"))
        if cell.server_config is None:
            return asyncio.run(
                replay_tasks(
                    tasks,
                    endpoints=cell.endpoints,
                    mode=cell.mode,
                    concurrency=cell.concurrency,
                    repetitions=cell.repetitions,
                    output_dir=output_dir,
                    request_overrides=cell.request_overrides,
                    request_timeout=request_timeout,
                    request_selection=cell.request_selection,
                    warmup=cell.warmup,
                    require_server_metrics=cell.framework == "minisgl",
                )
            )

        # A managed server is restarted for every measured repetition.  This makes each
        # repetition begin with an empty process-local KV cache instead of silently warming
        # later samples with the first repetition's requests.
        configs = load_server_configs(cell.server_config)
        records: list[dict[str, Any]] = []
        for repetition in range(1, cell.repetitions + 1):
            repetition_dir = output_dir / f"rep-{repetition:02d}"
            log_dir = output_dir / "server-logs" / f"rep-{repetition:02d}"
            log_dir.mkdir(parents=True)
            with ExitStack() as servers:
                for config in configs:
                    servers.enter_context(
                        running_server(
                            config,
                            log_dir / f"{config.config_id}.log",
                            startup_timeout,
                            shutdown_timeout,
                        )
                    )
                records.extend(
                    asyncio.run(
                        replay_tasks(
                            tasks,
                            endpoints=tuple(config.endpoint_url for config in configs),
                            mode=cell.mode,
                            concurrency=cell.concurrency,
                            repetitions=1,
                            repetition_start=repetition,
                            output_dir=repetition_dir,
                            request_overrides=cell.request_overrides,
                            request_timeout=request_timeout,
                            request_selection=cell.request_selection,
                            warmup=cell.warmup,
                            require_server_metrics=cell.framework == "minisgl",
                        )
                    )
                )
        records.sort(
            key=lambda item: (
                item["endpoint_index"],
                item["repetition"],
                item["case_id"],
                item["turn"],
            )
        )
        append_gzip_jsonl(output_dir / "result.jsonl.gz", records)
        (output_dir / "trajectory.txt").write_text(
            render_text_trajectory(records), encoding="utf-8"
        )
        return records


def run_experiment_manifest(
    manifest_path: Path,
    *,
    output_dir: Path,
    startup_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
) -> dict[str, Any]:
    roots, preferred, task_limit, cells = _load_manifest(manifest_path)
    tasks = _prepare_tasks(roots, preferred, task_limit)
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries: dict[str, Any] = {}
    records_by_cell: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        cell_dir = output_dir / cell.name
        records = _run_cell_replay(
            cell,
            tasks=tasks,
            output_dir=cell_dir,
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
            shutdown_timeout=shutdown_timeout,
        )
        summaries[cell.name] = summarize_records(records)
        records_by_cell[cell.name] = records
    comparisons: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    profiles: dict[str, Any] = {}
    for cell in cells:
        if cell.baseline_cell is None:
            continue
        if cell.baseline_cell not in records_by_cell:
            raise ValueError(f"unknown baseline_cell for {cell.name}: {cell.baseline_cell}")
        comparison = detect_slowdown(
            _cell_repetition_medians(records_by_cell[cell.name]),
            _cell_repetition_medians(records_by_cell[cell.baseline_cell]),
        )
        comparison["issue_attribution"] = attribute_issues(
            list(summaries[cell.name]["issues"]),
            list(summaries[cell.baseline_cell]["issues"]),
        )
        comparisons[cell.name] = comparison
        if comparison["profile_required"] and cell.profile_on_slowdown:
            if cell.mode == "rolling":
                diagnostic_cell = replace(
                    cell,
                    name=f"{cell.name}__rolling_drop_only",
                    mode="rolling-drop-only",
                    baseline_cell=cell.baseline_cell,
                    profile_on_slowdown=False,
                )
                diagnostic_records = _run_cell_replay(
                    diagnostic_cell,
                    tasks=tasks,
                    output_dir=output_dir / "diagnostics" / diagnostic_cell.name,
                    startup_timeout=startup_timeout,
                    request_timeout=request_timeout,
                    shutdown_timeout=shutdown_timeout,
                )
                diagnostic_summary = summarize_records(diagnostic_records)
                diagnostic_comparison = detect_slowdown(
                    _cell_repetition_medians(diagnostic_records),
                    _cell_repetition_medians(records_by_cell[cell.baseline_cell]),
                )
                diagnostic_comparison["issue_attribution"] = attribute_issues(
                    list(diagnostic_summary["issues"]),
                    list(summaries[cell.baseline_cell]["issues"]),
                )
                diagnostics[cell.name] = {
                    "cell": diagnostic_summary,
                    "comparison_to_baseline": diagnostic_comparison,
                }
            profiles[cell.name] = _run_profile_cell(
                cell,
                tasks=tasks,
                output_dir=output_dir / "profiles" / cell.name,
                startup_timeout=startup_timeout,
                request_timeout=request_timeout,
                shutdown_timeout=shutdown_timeout,
            )
    summary = {
        "manifest": str(manifest_path.resolve()),
        "task_ids": [task.case_id for task in tasks],
        "tasks": [
            {
                "case_id": task.case_id,
                "prompt_tokens_hint": task.prompt_tokens_hint,
                "request_count": len(task.requests),
                "provenance": sorted({request.provenance for request in task.requests}),
            }
            for task in tasks
        ],
        "cells": summaries,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "profiles": profiles,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def audit_request_matrix(
    matrix_dir: Path, *, require_server_metrics: bool = False
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for case_dir in sorted(matrix_dir.glob("request-*")):
        requests_path = case_dir / "requests.jsonl"
        results_path = case_dir / "results.jsonl"
        if not requests_path.is_file() or not results_path.is_file():
            continue
        requests = [json.loads(line) for line in requests_path.read_text().splitlines() if line]
        for line in results_path.read_text(encoding="utf-8").splitlines():
            result = json.loads(line)
            for response_item in result.get("response_chain", []):
                index = int(response_item["request_index"]) - 1
                response = response_item.get("response")
                parsed_response: dict[str, Any] = {"ok": False}
                audit = {"issues": ["transport:missing_response"], "inspection": {}}
                if isinstance(response, dict) and 0 <= index < len(requests):
                    try:
                        parsed = parse_response_bytes(
                            response.get("body_text", "").encode("utf-8"),
                            stream=requests[index].get("stream") is True,
                        )
                        parsed_response = {
                            "ok": response_item.get("outcome") == "success",
                            **parsed,
                        }
                        audit = audit_parsed_response(parsed, request=requests[index])
                        if require_server_metrics:
                            # A fixed matrix can reuse an exact key across non-stream/stream
                            # cases, and an effective Reposition can legitimately have no
                            # reusable old-position KV. Validate every stress request, then
                            # require positive Retry activity once per server configuration
                            # after the complete matrix has been collected.
                            audit["issues"] = sorted(
                                set(audit["issues"])
                                | set(
                                    inspect_metrics(
                                        parsed.get("server_metrics"),
                                        stress=bool(requests[index].get("reposition")),
                                        cold=False,
                                    )
                                )
                            )
                            metrics = parsed.get("server_metrics")
                            if isinstance(metrics, dict) and bool(
                                requests[index].get("reposition")
                            ):
                                transition = int(metrics.get("reposition_transition_count", 0))
                                h2d_bytes = int(metrics.get("reposition_h2d_bytes", 0))
                                if (transition > 0) != (h2d_bytes > 0):
                                    audit["issues"] = sorted(
                                        set(audit["issues"])
                                        | {"system:retry_transition_h2d_inconsistent"}
                                    )
                    except Exception as exc:
                        audit = {
                            "issues": ["transport:request_or_parse_failure"],
                            "inspection": {},
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        }
                records.append(
                    {
                        "config_name": result.get("config_name"),
                        "case_id": result.get("case_id"),
                        "mode": (
                            "stream"
                            if 0 <= index < len(requests) and requests[index].get("stream") is True
                            else "nonstream"
                        ),
                        "turn": index + 1,
                        "request_index": index + 1,
                        "response": parsed_response,
                        "audit": audit,
                        "reposition_stress": (
                            bool(requests[index].get("reposition"))
                            if 0 <= index < len(requests)
                            else False
                        ),
                    }
                )
    if require_server_metrics:
        by_config: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            if record["reposition_stress"] and record["response"].get("ok"):
                by_config.setdefault(str(record.get("config_name")), []).append(record)
        for config_records in by_config.values():
            complete_metrics = [
                record["response"].get("server_metrics")
                for record in config_records
                if isinstance(record["response"].get("server_metrics"), dict)
                and METRIC_FIELDS <= set(record["response"]["server_metrics"])
            ]
            if complete_metrics and not any(
                int(metrics["reposition_transition_count"]) > 0
                and int(metrics["reposition_h2d_bytes"]) > 0
                for metrics in complete_metrics
            ):
                first = config_records[0]
                first["audit"]["issues"] = sorted(
                    set(first["audit"]["issues"]) | {"system:retry_activity_missing"}
                )
    return {"records": records, "summary": summarize_records(records)}


class _ProxyState:
    def __init__(
        self,
        upstream: str,
        mode: str,
        audit_output: Path,
        *,
        require_server_metrics: bool = False,
    ) -> None:
        self.upstream = upstream.rstrip("/")
        self.mode = mode
        self.audit_output = audit_output
        self.require_server_metrics = require_server_metrics
        self.lock = threading.Lock()
        self.sequence = 0

    def next_sequence(self) -> int:
        with self.lock:
            self.sequence += 1
            return self.sequence

    def record(self, value: dict[str, Any]) -> None:
        with self.lock:
            append_gzip_jsonl(self.audit_output, [value])


def _audit_proxy_response(
    parsed: dict[str, Any],
    *,
    request: dict[str, Any],
    require_server_metrics: bool,
) -> dict[str, Any]:
    audit = audit_parsed_response(parsed, request=request)
    if require_server_metrics:
        has_reposition = bool(request.get("reposition"))
        audit["issues"] = sorted(
            set(audit["issues"])
            | set(
                inspect_metrics(
                    parsed.get("server_metrics"),
                    stress=has_reposition,
                    cold=has_reposition,
                )
            )
        )
    return audit


def make_proxy_handler(state: _ProxyState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *_: Any) -> None:
            return

        def _upstream_url(self) -> str:
            return state.upstream + self.path

        def _copy_response_headers(self, response: httpx.Response) -> None:
            for key, value in response.headers.multi_items():
                if key.lower() not in _HOP_HEADERS:
                    self.send_header(key, value)

        def do_GET(self) -> None:
            with httpx.stream(
                "GET", self._upstream_url(), timeout=1800, trust_env=False
            ) as response:
                self.send_response(response.status_code)
                self._copy_response_headers(response)
                self.end_headers()
                for chunk in response.iter_raw():
                    self.wfile.write(chunk)
                    self.wfile.flush()

        def do_POST(self) -> None:
            sequence = state.next_sequence()
            length = int(self.headers.get("Content-Length", "0"))
            original = json.loads(self.rfile.read(length))
            if not isinstance(original, dict):
                self.send_error(400, "request must be an object")
                return
            try:
                forwarded, plan = apply_rolling_interface(original, mode=state.mode)
            except ValueError as exc:
                self.send_error(400, str(exc))
                return
            request_headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in _HOP_HEADERS and key.lower() != "host"
            }
            raw = bytearray()
            started_ns = time.time_ns()
            with httpx.stream(
                "POST",
                self._upstream_url(),
                json=forwarded,
                headers=request_headers,
                timeout=1800,
                trust_env=False,
            ) as response:
                self.send_response(response.status_code)
                self._copy_response_headers(response)
                self.end_headers()
                for chunk in response.iter_raw():
                    raw.extend(chunk)
                    self.wfile.write(chunk)
                    self.wfile.flush()
                status_code = response.status_code
            record: dict[str, Any] = {
                "sequence": sequence,
                "mode": state.mode,
                "status_code": status_code,
                "started_ns": started_ns,
                "finished_ns": time.time_ns(),
                "request": forwarded,
                "rolling_plan": plan.to_dict(),
                "raw_response_hex": bytes(raw).hex(),
            }
            try:
                parsed = parse_response_bytes(bytes(raw), stream=forwarded.get("stream") is True)
                record["response"] = parsed
                record["audit"] = _audit_proxy_response(
                    parsed,
                    request=forwarded,
                    require_server_metrics=state.require_server_metrics,
                )
            except Exception as exc:
                record["audit"] = {
                    "issues": ["transport:request_or_parse_failure"],
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            state.record(record)

    return Handler


def serve_proxy(
    *,
    host: str,
    port: int,
    upstream: str,
    mode: str,
    audit_output: Path,
    require_server_metrics: bool = False,
) -> None:
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    state = _ProxyState(
        upstream,
        mode,
        audit_output,
        require_server_metrics=require_server_metrics,
    )
    server = ThreadingHTTPServer((host, port), make_proxy_handler(state))
    server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and audit Reposition experiment matrices.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--startup-timeout", type=float, default=1800)
    run.add_argument("--request-timeout", type=float, default=1800)
    run.add_argument("--shutdown-timeout", type=float, default=60)

    audit = subparsers.add_parser("audit-request-matrix")
    audit.add_argument("--matrix-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--trajectory-output", type=Path)
    audit.add_argument("--require-server-metrics", action="store_true")

    proxy = subparsers.add_parser("proxy")
    proxy.add_argument("--host", default="127.0.0.1")
    proxy.add_argument("--port", type=int, required=True)
    proxy.add_argument("--upstream", required=True)
    proxy.add_argument("--mode", choices=("full", "rolling", "rolling-drop-only"), required=True)
    proxy.add_argument("--audit-output", type=Path, required=True)
    proxy.add_argument("--require-server-metrics", action="store_true")

    fixed = subparsers.add_parser("materialize-fixed")
    fixed.add_argument("--capture-root", type=Path, action="append", required=True)
    fixed.add_argument("--preferred-case-id", action="append", default=[])
    fixed.add_argument("--task-limit", type=int, default=20)
    fixed.add_argument("--output-dir", type=Path, required=True)
    fixed.add_argument("--model")
    fixed.add_argument("--max-tokens", type=int, default=2048)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "proxy":
        serve_proxy(
            host=args.host,
            port=args.port,
            upstream=args.upstream,
            mode=args.mode,
            audit_output=args.audit_output,
            require_server_metrics=args.require_server_metrics,
        )
        return 0
    if args.command == "audit-request-matrix":
        report = audit_request_matrix(
            args.matrix_dir, require_server_metrics=args.require_server_metrics
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        trajectory_output = args.trajectory_output or args.output.with_suffix(".trajectory.txt")
        trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        trajectory_output.write_text(
            render_text_trajectory(report["records"]),
            encoding="utf-8",
        )
        return 0
    if args.command == "materialize-fixed":
        tasks = _prepare_tasks(args.capture_root, args.preferred_case_id, args.task_limit)
        paths = materialize_fixed_request_cases(
            tasks,
            args.output_dir,
            model=args.model,
            max_tokens=args.max_tokens,
        )
        print(f"Wrote {len(paths)} fixed request cases to {args.output_dir}")
        return 0
    summary = run_experiment_manifest(
        args.manifest,
        output_dir=args.output_dir,
        startup_timeout=args.startup_timeout,
        request_timeout=args.request_timeout,
        shutdown_timeout=args.shutdown_timeout,
    )
    return 0 if all(not cell["issues"] for cell in summary["cells"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
