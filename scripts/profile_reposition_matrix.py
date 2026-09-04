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
from dataclasses import asdict, dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Iterable, Sequence

PROFILE_CONFIG_ENV = "MINISGL_R10_PROFILE_CONFIG"


@dataclass(frozen=True)
class ProfileTarget:
    stage: str
    filename_suffix: str
    qualname: str
    phase: str | None = None


MINISGL_TARGETS = (
    ProfileTarget("tokenize", "minisgl/tokenizer/tokenize.py", "TokenizeManager.tokenize"),
    ProfileTarget("scheduler", "minisgl/scheduler/scheduler.py", "Scheduler._process_one_msg"),
    ProfileTarget("scheduler", "minisgl/scheduler/scheduler.py", "Scheduler._schedule_next_batch"),
    ProfileTarget("prefill_extend", "minisgl/scheduler/prefill.py", "PrefillAdder.try_add_one"),
    ProfileTarget(
        "prefill_extend", "minisgl/engine/engine.py", "Engine.forward_batch", phase="prefill"
    ),
    ProfileTarget("decode", "minisgl/engine/engine.py", "Engine.forward_batch", phase="decode"),
    ProfileTarget("decode", "minisgl/scheduler/decode.py", "DecodeManager.schedule_next_batch"),
    ProfileTarget("radix_match", "minisgl/scheduler/cache.py", "CacheManager.match_req"),
    ProfileTarget("free_and_cache", "minisgl/scheduler/cache.py", "CacheManager.cache_req"),
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

    def __init__(self, targets: Sequence[ProfileTarget], output: Path, *, nvtx: bool) -> None:
        self.targets = tuple(targets)
        self.output = output
        self.nvtx = nvtx
        self._local = threading.local()
        self._lock = threading.Lock()
        self._aggregates: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
        self._nvtx_module: Any | None = None

    def _match(self, frame: FrameType) -> ProfileTarget | None:
        filename = frame.f_code.co_filename.replace("\\", "/")
        qualname = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
        for target in self.targets:
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
        stacks = getattr(self._local, "stacks", None)
        if stacks is None:
            stacks = self._local.stacks = {}
        frame_id = id(frame)
        if event == "call":
            target = self._match(frame)
            if target is None:
                return
            nvtx = self._nvtx()
            if nvtx is not None:
                nvtx.range_push(f"r10:{target.stage}:{target.qualname}")
            stacks[frame_id] = (target, time.perf_counter_ns(), nvtx is not None)
            return
        if event not in {"return", "exception"}:
            return
        active = stacks.pop(frame_id, None)
        if active is None:
            return
        target, started_ns, pushed = active
        elapsed_ns = time.perf_counter_ns() - started_ns
        if pushed:
            nvtx = self._nvtx()
            if nvtx is not None:
                nvtx.range_pop()
        qualname = target.qualname + (f"[{target.phase}]" if target.phase is not None else "")
        key = (target.stage, target.filename_suffix, qualname)
        with self._lock:
            aggregate = self._aggregates[key]
            aggregate[0] += 1
            aggregate[1] += elapsed_ns

    def install(self) -> None:
        sys.setprofile(self.callback)
        threading.setprofile(self.callback)
        atexit.register(self.flush)

    def flush(self) -> None:
        sys.setprofile(None)
        threading.setprofile(None)
        rows = [
            {
                "pid": os.getpid(),
                "stage": stage,
                "filename_suffix": filename,
                "qualname": qualname,
                "calls": value[0],
                "total_ns": value[1],
            }
            for (stage, filename, qualname), value in sorted(self._aggregates.items())
        ]
        if not rows:
            return
        output = Path(str(self.output).format(pid=os.getpid()))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")


_ACTIVE_PROFILER: EventProfiler | None = None


def install_from_env() -> EventProfiler | None:
    global _ACTIVE_PROFILER
    config_path = os.environ.get(PROFILE_CONFIG_ENV)
    if not config_path or _ACTIVE_PROFILER is not None:
        return _ACTIVE_PROFILER
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    targets = tuple(ProfileTarget(**item) for item in config["targets"])
    profiler = EventProfiler(targets, Path(config["output"]), nvtx=bool(config.get("nvtx", True)))
    profiler.install()
    _ACTIVE_PROFILER = profiler
    return profiler


def create_profile_bootstrap(
    output_dir: Path,
    *,
    framework: str,
    nvtx: bool = True,
) -> dict[str, str]:
    """Create a task-local sitecustomize and return env additions for every server child."""

    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "profile-config.json"
    config = {
        "framework": framework,
        "targets": [asdict(target) for target in targets_for_framework(framework)],
        "output": str((output_dir / "profile-{pid}.jsonl").resolve()),
        "nvtx": nvtx,
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


def summarize_profile(paths: Iterable[Path]) -> dict[str, Any]:
    stages: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "total_ns": 0})
    functions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            stage = stages[str(row["stage"])]
            stage["calls"] += int(row["calls"])
            stage["total_ns"] += int(row["total_ns"])
            key = (str(row["stage"]), str(row["filename_suffix"]), str(row["qualname"]))
            aggregate = functions.setdefault(
                key,
                {
                    "stage": key[0],
                    "filename_suffix": key[1],
                    "qualname": key[2],
                    "calls": 0,
                    "total_ns": 0,
                },
            )
            aggregate["calls"] += int(row["calls"])
            aggregate["total_ns"] += int(row["total_ns"])
    return {
        "stages": {
            key: {
                **value,
                "total_ms": value["total_ns"] / 1_000_000,
                "mean_ms": value["total_ns"] / value["calls"] / 1_000_000,
            }
            for key, value in sorted(stages.items())
        },
        "functions": [
            {
                **value,
                "total_ms": value["total_ns"] / 1_000_000,
                "mean_ms": value["total_ns"] / value["calls"] / 1_000_000,
            }
            for _, value in sorted(functions.items())
        ],
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
            args.output_dir, framework=args.framework, nvtx=not args.no_nvtx
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
