from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import statistics
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any, Iterable, Sequence

PROFILE_CONFIG_ENV = "MINISGL_R10_PROFILE_CONFIG"
DEFAULT_SNAPSHOT_INTERVAL_S = 5.0


@dataclass(frozen=True)
class ProfileTarget:
    stage: str
    filename_suffix: str
    qualname: str
    phase: str | None = None


MINISGL_TARGETS = (
    ProfileTarget("tokenize", "minisgl/tokenizer/tokenize.py", "TokenizeManager.tokenize"),
    ProfileTarget(
        "reposition_sequence",
        "minisgl/tokenizer/reposition_sequence.py",
        "RepositionSequenceState.open_msg",
    ),
    ProfileTarget(
        "reposition_sequence",
        "minisgl/tokenizer/reposition_sequence.py",
        "RepositionSequenceState.activate",
    ),
    ProfileTarget(
        "reposition_sequence",
        "minisgl/tokenizer/reposition_sequence.py",
        "RepositionSequenceState.build_next_msg",
    ),
    ProfileTarget(
        "reposition_sequence",
        "minisgl/tokenizer/reposition_sequence.py",
        "RepositionSequenceState.accept_ack",
    ),
    ProfileTarget(
        "reposition_sequence",
        "minisgl/scheduler/reposition_sequence.py",
        "SchedulerRepositionSequence.from_open",
    ),
    ProfileTarget(
        "reposition_sequence",
        "minisgl/scheduler/reposition_sequence.py",
        "SchedulerRepositionSequence.materialize",
    ),
    ProfileTarget("serialization", "minisgl/message/backend.py", "BaseBackendMsg.encoder"),
    ProfileTarget("serialization", "minisgl/message/backend.py", "BaseBackendMsg.decoder"),
    ProfileTarget("scheduler_loop", "minisgl/scheduler/scheduler.py", "Scheduler.overlap_loop"),
    ProfileTarget("scheduler_loop", "minisgl/scheduler/scheduler.py", "Scheduler.normal_loop"),
    ProfileTarget("scheduler", "minisgl/scheduler/scheduler.py", "Scheduler._process_one_msg"),
    ProfileTarget("scheduler", "minisgl/scheduler/scheduler.py", "Scheduler._schedule_next_batch"),
    ProfileTarget(
        "scheduler_result",
        "minisgl/scheduler/scheduler.py",
        "Scheduler._process_last_data",
    ),
    ProfileTarget(
        "scheduler_prepare", "minisgl/scheduler/scheduler.py", "Scheduler._prepare_batch"
    ),
    ProfileTarget("scheduler_forward", "minisgl/scheduler/scheduler.py", "Scheduler._forward"),
    ProfileTarget("scheduler_prepare", "minisgl/scheduler/scheduler.py", "_make_positions"),
    ProfileTarget("scheduler_prepare", "minisgl/scheduler/scheduler.py", "_make_input_tuple"),
    ProfileTarget("scheduler_prepare", "minisgl/scheduler/scheduler.py", "_make_write_tuple"),
    ProfileTarget("scheduler_idle", "minisgl/scheduler/scheduler.py", "Scheduler.run_when_idle"),
    ProfileTarget(
        "scheduler_receive", "minisgl/scheduler/io.py", "SchedulerIOMixin._recv_msg_single_rank"
    ),
    ProfileTarget(
        "scheduler_receive", "minisgl/scheduler/io.py", "SchedulerIOMixin._recv_msg_multi_rank0"
    ),
    ProfileTarget(
        "scheduler_receive", "minisgl/scheduler/io.py", "SchedulerIOMixin._recv_msg_multi_rank1"
    ),
    ProfileTarget(
        "scheduler_reply", "minisgl/scheduler/io.py", "SchedulerIOMixin._reply_tokenizer_rank0"
    ),
    ProfileTarget("scheduler_host", "minisgl/core.py", "Req.append_host"),
    ProfileTarget("scheduler_host", "minisgl/core.py", "Req.match_stop"),
    ProfileTarget(
        "scheduler_metrics",
        "minisgl/message/metrics.py",
        "RequestMetricsState.observe_token",
    ),
    ProfileTarget("scheduler_metrics", "minisgl/message/metrics.py", "RequestMetricsState.finish"),
    ProfileTarget("scheduler_ipc", "minisgl/utils/mp.py", "ZmqPushQueue.put"),
    ProfileTarget("scheduler_ipc", "minisgl/utils/mp.py", "ZmqPullQueue.get"),
    ProfileTarget("scheduler_ipc", "minisgl/utils/mp.py", "ZmqPullQueue.get_raw"),
    ProfileTarget("scheduler_ipc", "minisgl/utils/mp.py", "ZmqPullQueue.decode"),
    ProfileTarget("scheduler_ipc", "minisgl/utils/mp.py", "ZmqPubQueue.put_raw"),
    ProfileTarget("scheduler_ipc", "minisgl/utils/mp.py", "ZmqSubQueue.get"),
    ProfileTarget("prefill_extend", "minisgl/scheduler/prefill.py", "PrefillAdder.try_add_one"),
    ProfileTarget(
        "prefill_extend",
        "minisgl/scheduler/prefill.py",
        "PrefillAdder.plan_context_prefill",
    ),
    ProfileTarget(
        "prefill_extend", "minisgl/engine/engine.py", "Engine.forward_batch", phase="prefill"
    ),
    ProfileTarget("decode", "minisgl/engine/engine.py", "Engine.forward_batch", phase="decode"),
    ProfileTarget("decode", "minisgl/scheduler/decode.py", "DecodeManager.schedule_next_batch"),
    ProfileTarget("radix_match", "minisgl/scheduler/cache.py", "CacheManager.match_req"),
    ProfileTarget(
        "radix_match",
        "minisgl/scheduler/cache.py",
        "CacheManager.match_occurrence_req",
    ),
    ProfileTarget("radix_match", "minisgl/scheduler/cache.py", "CacheManager._match_req"),
    ProfileTarget("radix_match", "minisgl/scheduler/cache.py", "CacheManager._derive_active_match"),
    ProfileTarget("radix_match", "minisgl/kvcache/radix_cache.py", "RadixPrefixCache.match_prefix"),
    ProfileTarget("radix_match", "minisgl/kvcache/radix_cache.py", "RadixPrefixCache._tree_walk"),
    ProfileTarget(
        "radix_match", "minisgl/kvcache/radix_cache.py", "RadixTreeNode.find_exact_child"
    ),
    ProfileTarget("radix_match", "minisgl/kvcache/radix_cache.py", "RadixTreeNode.get_match_len"),
    ProfileTarget("radix_compare", "minisgl/kernel/radix.py", "fast_compare_radix_records"),
    ProfileTarget("radix_compare", "minisgl/kernel/radix.py", "radix_record_edge_hash"),
    ProfileTarget("radix_compare", "minisgl/kernel/radix.py", "radix_record_edge_equal"),
    ProfileTarget(
        "radix_compile",
        "minisgl/kernel/radix_reposition.py",
        "compile_radix_reposition_layout",
    ),
    ProfileTarget(
        "radix_compile",
        "minisgl/kernel/radix_reposition.py",
        "compile_radix_reposition_layout_batch",
    ),
    ProfileTarget("free_and_cache", "minisgl/scheduler/cache.py", "CacheManager.cache_req"),
    ProfileTarget(
        "free_and_cache",
        "minisgl/scheduler/cache.py",
        "CacheManager._cache_finished_delta_req",
    ),
    ProfileTarget(
        "free_and_cache",
        "minisgl/scheduler/cache.py",
        "CacheManager._free_finished_candidates",
    ),
    ProfileTarget(
        "radix_insert", "minisgl/kvcache/radix_cache.py", "RadixPrefixCache.insert_prefix"
    ),
    ProfileTarget("radix_insert", "minisgl/kvcache/radix_cache.py", "RadixPrefixCache._split_node"),
    ProfileTarget(
        "radix_insert",
        "minisgl/kvcache/radix_cache.py",
        "RadixPrefixCache._register_ordinary_node",
    ),
    ProfileTarget("radix_insert", "minisgl/kvcache/radix_cache.py", "RadixTreeNode.set_key_value"),
    ProfileTarget("integrity", "minisgl/scheduler/cache.py", "CacheManager.check_integrity"),
    ProfileTarget(
        "integrity", "minisgl/kvcache/radix_cache.py", "RadixPrefixCache.check_integrity"
    ),
    ProfileTarget("allocation", "minisgl/scheduler/cache.py", "CacheManager._allocate"),
    ProfileTarget("allocation", "minisgl/scheduler/cache.py", "CacheManager._free"),
    ProfileTarget("evict", "minisgl/kvcache/radix_cache.py", "RadixPrefixCache.evict"),
    ProfileTarget(
        "evict",
        "minisgl/kvcache/radix_cache.py",
        "RadixPrefixCache._collect_leave_nodes_for_evict",
    ),
)

SGLANG_TARGETS = (
    ProfileTarget(
        "tokenize", "srt/managers/tokenizer_manager.py", "TokenizerManager.generate_request"
    ),
    ProfileTarget("tokenize", "srt/managers/tokenizer_manager.py", "TokenizerManager.handle_loop"),
    ProfileTarget("scheduler", "srt/managers/scheduler.py", "Scheduler.process_input_requests"),
    ProfileTarget("scheduler", "srt/managers/scheduler.py", "Scheduler.get_next_batch_to_run"),
    ProfileTarget("scheduler", "srt/managers/scheduler.py", "Scheduler.run_batch"),
    ProfileTarget("scheduler", "srt/managers/scheduler.py", "Scheduler.process_batch_result"),
    ProfileTarget(
        "prefill_extend", "srt/managers/schedule_batch.py", "ScheduleBatch.prepare_for_extend"
    ),
    ProfileTarget("decode", "srt/managers/schedule_batch.py", "ScheduleBatch.prepare_for_decode"),
    ProfileTarget(
        "prefill_extend", "srt/model_executor/model_runner.py", "ModelRunner.forward_extend"
    ),
    ProfileTarget("decode", "srt/model_executor/model_runner.py", "ModelRunner.forward_decode"),
    ProfileTarget("radix_match", "srt/mem_cache/radix_cache.py", "RadixCache.match_prefix"),
    ProfileTarget(
        "free_and_cache", "srt/mem_cache/radix_cache.py", "RadixCache.cache_finished_req"
    ),
    ProfileTarget(
        "free_and_cache", "srt/mem_cache/radix_cache.py", "RadixCache.cache_unfinished_req"
    ),
    ProfileTarget("evict", "srt/mem_cache/radix_cache.py", "RadixCache.evict"),
)


def targets_for_framework(framework: str) -> tuple[ProfileTarget, ...]:
    if framework == "minisgl":
        return MINISGL_TARGETS
    if framework == "sglang":
        return SGLANG_TARGETS
    raise ValueError(f"unknown framework: {framework}")


class EventProfiler:
    """Low-intrusion matched-function timing with optional unsynchronized NVTX ranges."""

    def __init__(
        self,
        targets: Sequence[ProfileTarget],
        output: Path,
        *,
        nvtx: bool,
        sample_limit: int = 4096,
        activation_file: Path | None = None,
        snapshot_interval_s: float = DEFAULT_SNAPSHOT_INTERVAL_S,
    ) -> None:
        if sample_limit < 1:
            raise ValueError("sample_limit must be positive")
        if snapshot_interval_s <= 0:
            raise ValueError("snapshot_interval_s must be positive")
        self.targets = tuple(targets)
        self.output = output
        self.nvtx = nvtx
        self.sample_limit = sample_limit
        self.activation_file = activation_file
        self.snapshot_interval_s = snapshot_interval_s
        self._enabled = activation_file is None
        self._activated_ns = time.perf_counter_ns() if self._enabled else None
        self._stop = threading.Event()
        self._snapshot_write_lock = threading.Lock()
        self._monitor: threading.Thread | None = None
        self._local = threading.local()
        self._lock = threading.Lock()
        self._targets_by_name: dict[str, list[ProfileTarget]] = defaultdict(list)
        for target in self.targets:
            self._targets_by_name[target.qualname.rsplit(".", 1)[-1]].append(target)
        self._aggregates: dict[tuple[str, str, str], _ProfileAggregate] = defaultdict(
            _ProfileAggregate
        )
        self._random = random.Random(17)
        self._nvtx_module: Any | None = None

    def _match(self, frame: FrameType) -> ProfileTarget | None:
        filename = frame.f_code.co_filename.replace("\\", "/")
        qualname = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
        for target in self._targets_by_name.get(frame.f_code.co_name, ()):
            if not filename.endswith(target.filename_suffix) or not qualname.endswith(
                target.qualname
            ):
                continue
            if target.phase is not None:
                batch = frame.f_locals.get("batch")
                if batch is None:
                    forward_input = frame.f_locals.get("forward_input")
                    batch = getattr(forward_input, "batch", None)
                if getattr(batch, "phase", None) != target.phase:
                    continue
            return target
        return None

    def _nvtx(self) -> Any | None:
        if not self.nvtx:
            return None
        if self._nvtx_module is None:
            try:
                import torch.cuda.nvtx as nvtx_module
            except (ImportError, RuntimeError):
                self.nvtx = False
                return None
            self._nvtx_module = nvtx_module
        return self._nvtx_module

    def callback(self, frame: FrameType, event: str, _: Any) -> None:
        if not self._enabled:
            return
        active = getattr(self._local, "active", None)
        if active is None:
            active = self._local.active = []
        frames = getattr(self._local, "frames", None)
        if frames is None:
            frames = self._local.frames = {}
        frame_id = id(frame)
        if event == "call":
            target = self._match(frame)
            if target is None:
                return
            nvtx = self._nvtx()
            if nvtx is not None:
                nvtx.range_push(f"r10:{target.stage}:{target.qualname}")
            entry = _ActiveCall(frame_id, target, time.perf_counter_ns(), 0, nvtx is not None)
            active.append(entry)
            frames[frame_id] = entry
            return
        if event not in {"return", "exception"}:
            return
        entry = frames.pop(frame_id, None)
        if entry is None:
            return
        if not active or active[-1] is not entry:
            return
        active.pop()
        elapsed_ns = time.perf_counter_ns() - entry.started_ns
        self_ns = max(0, elapsed_ns - entry.profiled_child_ns)
        if active:
            active[-1].profiled_child_ns += elapsed_ns
        if entry.nvtx_pushed:
            nvtx = self._nvtx()
            if nvtx is not None:
                nvtx.range_pop()
        target = entry.target
        qualname = target.qualname + (f"[{target.phase}]" if target.phase is not None else "")
        key = (target.stage, target.filename_suffix, qualname)
        with self._lock:
            aggregate = self._aggregates[key]
            aggregate.add(
                elapsed_ns,
                self_ns,
                sample_limit=self.sample_limit,
                generator=self._random,
            )

    def install(self) -> None:
        sys.setprofile(self.callback)
        threading.setprofile(self.callback)
        self._monitor = threading.Thread(
            target=self._monitor_activation_and_snapshot,
            name="minisgl-profile-snapshot",
            daemon=True,
        )
        self._monitor.start()
        atexit.register(self.flush)

    def activate(self) -> None:
        if not self._enabled:
            self._activated_ns = time.perf_counter_ns()
            self._enabled = True

    def _monitor_activation_and_snapshot(self) -> None:
        while not self._stop.wait(0.05 if not self._enabled else self.snapshot_interval_s):
            if not self._enabled:
                if self.activation_file is not None and self.activation_file.exists():
                    self.activate()
                continue
            self.snapshot()

    def snapshot(self) -> None:
        with self._lock:
            rows = [
                {
                    "pid": os.getpid(),
                    "activated_ns": self._activated_ns,
                    "stage": stage,
                    "filename_suffix": filename,
                    "qualname": qualname,
                    **value.to_dict(),
                }
                for (stage, filename, qualname), value in sorted(self._aggregates.items())
            ]
        if not rows:
            return
        output = Path(str(self.output).format(pid=os.getpid()))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        with self._snapshot_write_lock:
            temporary.write_text(payload, encoding="utf-8", newline="\n")
            temporary.replace(output)

    def flush(self) -> None:
        sys.setprofile(None)
        threading.setprofile(None)
        self._stop.set()
        self.snapshot()


@dataclass
class _ActiveCall:
    frame_id: int
    target: ProfileTarget
    started_ns: int
    profiled_child_ns: int
    nvtx_pushed: bool


@dataclass
class _ProfileAggregate:
    calls: int = 0
    total_ns: int = 0
    self_ns: int = 0
    min_ns: int | None = None
    max_ns: int = 0
    samples_ns: list[int] = field(default_factory=list)
    self_samples_ns: list[int] = field(default_factory=list)

    def add(
        self,
        elapsed_ns: int,
        self_ns: int,
        *,
        sample_limit: int,
        generator: random.Random,
    ) -> None:
        self.calls += 1
        self.total_ns += elapsed_ns
        self.self_ns += self_ns
        self.min_ns = elapsed_ns if self.min_ns is None else min(self.min_ns, elapsed_ns)
        self.max_ns = max(self.max_ns, elapsed_ns)
        if len(self.samples_ns) < sample_limit:
            self.samples_ns.append(elapsed_ns)
            self.self_samples_ns.append(self_ns)
            return
        replacement = generator.randrange(self.calls)
        if replacement < sample_limit:
            self.samples_ns[replacement] = elapsed_ns
            self.self_samples_ns[replacement] = self_ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "total_ns": self.total_ns,
            "self_ns": self.self_ns,
            "min_ns": self.min_ns or 0,
            "max_ns": self.max_ns,
            "samples_ns": self.samples_ns,
            "self_samples_ns": self.self_samples_ns,
        }


_ACTIVE_PROFILER: EventProfiler | None = None


def install_from_env() -> EventProfiler | None:
    global _ACTIVE_PROFILER
    config_path = os.environ.get(PROFILE_CONFIG_ENV)
    if not config_path or _ACTIVE_PROFILER is not None:
        return _ACTIVE_PROFILER
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    targets = tuple(ProfileTarget(**item) for item in config["targets"])
    activation_file = config.get("activation_file")
    profiler = EventProfiler(
        targets,
        Path(config["output"]),
        nvtx=bool(config.get("nvtx", True)),
        activation_file=Path(activation_file) if activation_file else None,
        snapshot_interval_s=float(config.get("snapshot_interval_s", DEFAULT_SNAPSHOT_INTERVAL_S)),
    )
    profiler.install()
    _ACTIVE_PROFILER = profiler
    return profiler


def create_profile_bootstrap(
    output_dir: Path,
    *,
    framework: str,
    nvtx: bool = True,
    activation_file: Path | None = None,
    snapshot_interval_s: float = DEFAULT_SNAPSHOT_INTERVAL_S,
) -> dict[str, str]:
    """Create a task-local sitecustomize and return env additions for every server child."""

    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "profile-config.json"
    config = {
        "framework": framework,
        "targets": [asdict(target) for target in targets_for_framework(framework)],
        "output": str((output_dir / "profile-{pid}.jsonl").resolve()),
        "nvtx": nvtx,
        "activation_file": str(activation_file.resolve()) if activation_file else None,
        "snapshot_interval_s": snapshot_interval_s,
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bootstrap = output_dir / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "sitecustomize.py").write_text(
        "from scripts.profile_reposition_matrix import install_from_env\ninstall_from_env()\n",
        encoding="utf-8",
    )
    current_pythonpath = os.environ.get("PYTHONPATH")
    project_root = Path(__file__).resolve().parents[1]
    pythonpath = os.pathsep.join((str(bootstrap.resolve()), str(project_root)))
    if current_pythonpath:
        pythonpath += os.pathsep + current_pythonpath
    return {PROFILE_CONFIG_ENV: str(config_path.resolve()), "PYTHONPATH": pythonpath}


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _timing_summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    calls = int(aggregate["calls"])
    total_ns = int(aggregate["total_ns"])
    self_ns = int(aggregate["self_ns"])
    samples_ns = aggregate["samples_ns"]
    self_samples_ns = aggregate["self_samples_ns"]
    return {
        **aggregate,
        "total_ms": total_ns / 1_000_000,
        "self_ms": self_ns / 1_000_000,
        "mean_ms": total_ns / calls / 1_000_000,
        "self_mean_ms": self_ns / calls / 1_000_000,
        "p50_ms": _percentile(samples_ns, 0.5) / 1_000_000,
        "p95_ms": _percentile(samples_ns, 0.95) / 1_000_000,
        "self_p50_ms": _percentile(self_samples_ns, 0.5) / 1_000_000,
        "self_p95_ms": _percentile(self_samples_ns, 0.95) / 1_000_000,
        "min_ms": int(aggregate["min_ns"]) / 1_000_000,
        "max_ms": int(aggregate["max_ns"]) / 1_000_000,
        "sample_count": len(samples_ns),
    }


def summarize_profile(paths: Iterable[Path]) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "total_ns": 0,
            "self_ns": 0,
            "min_ns": 0,
            "max_ns": 0,
            "samples_ns": [],
            "self_samples_ns": [],
        }
    )
    functions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            stage = stages[str(row["stage"])]
            stage["calls"] += int(row["calls"])
            stage["total_ns"] += int(row["total_ns"])
            stage["self_ns"] += int(row.get("self_ns", row["total_ns"]))
            row_min = int(row.get("min_ns", 0))
            if row_min and (not stage["min_ns"] or row_min < stage["min_ns"]):
                stage["min_ns"] = row_min
            stage["max_ns"] = max(stage["max_ns"], int(row.get("max_ns", 0)))
            stage["samples_ns"].extend(int(value) for value in row.get("samples_ns", []))
            stage["self_samples_ns"].extend(int(value) for value in row.get("self_samples_ns", []))
            key = (str(row["stage"]), str(row["filename_suffix"]), str(row["qualname"]))
            aggregate = functions.setdefault(
                key,
                {
                    "stage": key[0],
                    "filename_suffix": key[1],
                    "qualname": key[2],
                    "calls": 0,
                    "total_ns": 0,
                    "self_ns": 0,
                    "min_ns": 0,
                    "max_ns": 0,
                    "samples_ns": [],
                    "self_samples_ns": [],
                },
            )
            aggregate["calls"] += int(row["calls"])
            aggregate["total_ns"] += int(row["total_ns"])
            aggregate["self_ns"] += int(row.get("self_ns", row["total_ns"]))
            if row_min and (not aggregate["min_ns"] or row_min < aggregate["min_ns"]):
                aggregate["min_ns"] = row_min
            aggregate["max_ns"] = max(aggregate["max_ns"], int(row.get("max_ns", 0)))
            aggregate["samples_ns"].extend(int(value) for value in row.get("samples_ns", []))
            aggregate["self_samples_ns"].extend(
                int(value) for value in row.get("self_samples_ns", [])
            )
    return {
        "stages": {key: _timing_summary(value) for key, value in sorted(stages.items())},
        "functions": [_timing_summary(value) for _, value in sorted(functions.items())],
    }


def bootstrap_ratio_interval(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 17,
) -> tuple[float, float]:
    if not candidate or not baseline or samples < 1:
        raise ValueError("candidate, baseline, and samples must be non-empty/positive")
    generator = random.Random(seed)
    ratios: list[float] = []
    for _ in range(samples):
        candidate_sample = [generator.choice(candidate) for _ in candidate]
        baseline_sample = [generator.choice(baseline) for _ in baseline]
        denominator = statistics.median(baseline_sample)
        if denominator <= 0:
            raise ValueError("baseline samples must be positive")
        ratios.append(statistics.median(candidate_sample) / denominator)
    ratios.sort()
    return ratios[int(0.025 * (samples - 1))], ratios[int(0.975 * (samples - 1))]


def detect_slowdown(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    threshold: float = 1.15,
) -> dict[str, Any]:
    if len(candidate) != len(baseline):
        raise ValueError("paired candidate and baseline repetitions must have equal lengths")
    if len(candidate) < 3 or any(value <= 0 for value in baseline):
        raise ValueError("at least three positive paired repetitions are required")
    paired = [left / right for left, right in zip(candidate, baseline)]
    lower, upper = bootstrap_ratio_interval(candidate, baseline)
    ratio = statistics.median(candidate) / statistics.median(baseline)
    repeated = sum(item > threshold for item in paired) >= 2
    return {
        "candidate_median": statistics.median(candidate),
        "baseline_median": statistics.median(baseline),
        "ratio": ratio,
        "paired_ratios": paired,
        "confidence_interval_95": [lower, upper],
        "threshold": threshold,
        "repeated_2_of_3": repeated,
        "profile_required": ratio > threshold and repeated and lower > 1.0,
        "severe_2x": ratio >= 2.0,
        "severe_4x": ratio >= 4.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile and compare Reposition matrix stages.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    install = subparsers.add_parser("create-bootstrap")
    install.add_argument("--output-dir", type=Path, required=True)
    install.add_argument("--framework", choices=("minisgl", "sglang"), required=True)
    install.add_argument("--no-nvtx", action="store_true")
    install.add_argument("--activation-file", type=Path)
    install.add_argument("--snapshot-interval-s", type=float, default=DEFAULT_SNAPSHOT_INTERVAL_S)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--inputs", type=Path, nargs="+", required=True)
    summarize.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--candidate", type=float, nargs="+", required=True)
    compare.add_argument("--baseline", type=float, nargs="+", required=True)
    compare.add_argument("--threshold", type=float, default=1.15)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create-bootstrap":
        additions = create_profile_bootstrap(
            args.output_dir,
            framework=args.framework,
            nvtx=not args.no_nvtx,
            activation_file=args.activation_file,
            snapshot_interval_s=args.snapshot_interval_s,
        )
        print(json.dumps(additions, sort_keys=True))
        return 0
    if args.command == "summarize":
        report = summarize_profile(args.inputs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0
    report = detect_slowdown(args.candidate, args.baseline, threshold=args.threshold)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["profile_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
