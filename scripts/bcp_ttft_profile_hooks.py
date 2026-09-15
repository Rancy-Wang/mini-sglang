"""Process-local experiment hooks. Never imported by production entry points.

PLAN-CS-20260915-R1: buffered, targeted timings; no global sys.setprofile.
Wall time is NOT GPU kernel time. CPU time is the calling thread's CPU time.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import json
import os
import resource
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path


class WaveGate:
    def __init__(self):
        self.wave = None
        self.expected = 1
        self.arrivals = set()
        self.first_ns = None
        self.released = False

    def arrive(self, wave, expected, uid):
        if wave != self.wave:
            if self.wave is not None and not self.released:
                raise RuntimeError("Previous wave was not released")
            self.wave, self.expected = wave, expected
            self.arrivals = set()
            self.first_ns = time.perf_counter_ns()
            self.released = False
        self.arrivals.add(uid)
        if len(self.arrivals) > expected:
            raise RuntimeError("More UIDs than the approved wave size")

    def ready(self):
        return len(self.arrivals) == self.expected


def replace_callable(owner, name, wrapper):
    """Preserve binding semantics for static/class/ordinary methods."""
    descriptor = inspect.getattr_static(owner, name)
    fn = descriptor.__func__ if isinstance(descriptor, (staticmethod, classmethod)) else descriptor
    replacement = wrapper(fn)
    setattr(owner, name, type(descriptor)(replacement)
            if isinstance(descriptor, (staticmethod, classmethod)) else replacement)
    return fn, replacement


def gpu_detail_enabled(control, phase):
    # Existing long-context and final turns; no added request or kernel sync.
    return bool(control.get("gpu_detail") and control.get("turn") in (6, 11) and phase == "prefill")


def communication_config(config, timeout):
    """Override only the experiment's frozen config, leaving defaults untouched."""
    if timeout <= 0:
        raise ValueError("Communication timeout must be positive")
    return replace(config, distributed_timeout=timeout)


def install():
    if getattr(install, "done", False):
        return
    install.done = True
    import torch
    from minisgl.engine.engine import Engine
    from minisgl.message import UserMsg
    from minisgl.scheduler.scheduler import Scheduler
    from minisgl.tokenizer.tokenize import TokenizeManager
    from scripts.profile_reposition_matrix import MINISGL_TARGETS

    root = Path(os.environ["MINISGL_TTFT_PROFILE_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    local = threading.local()
    events, pending_gpu, pending_ranges = [], [], []
    gate = WaveGate()
    current = {}
    cleared = set()
    batch_number = 0

    def emit(kind, **values):
        events.append(dict(kind=kind, pid=os.getpid(), tid=threading.get_native_id(),
                           cell=current.get("cell"), turn=current.get("turn"), **values))

    def control():
        current.clear()
        current.update(json.loads((root / "control.json").read_text()))

    original_communication = Engine._init_communication

    def init_communication(self, config):
        timeout = float(os.environ.get("MINISGL_TTFT_DISTRIBUTED_TIMEOUT", "600"))
        emit("communication_config", rank=config.tp_info.rank,
             original_timeout_s=config.distributed_timeout, effective_timeout_s=timeout)
        return original_communication(self, communication_config(config, timeout))

    Engine._init_communication = init_communication

    def flush():
        for row, start, stop in pending_ranges:
            if stop.query():
                row["gpu_stream_ms"] = start.elapsed_time(stop)
                events.append(row)
        pending_ranges[:] = [x for x in pending_ranges if "gpu_stream_ms" not in x[0]]
        for row, start, stop, output in pending_gpu:
            if stop is not None and not stop.query():
                continue
            if not output.copy_done_event.query():
                continue
            row["tokens"] = output.next_tokens_cpu.tolist()
            if stop is not None:
                row["gpu_stream_ms"] = start.elapsed_time(stop)
            events.append(row)
        pending_gpu[:] = [x for x in pending_gpu if "tokens" not in x[0]]
        if events:
            with (root / f"events-{os.getpid()}.jsonl").open("a") as stream:
                stream.write("".join(json.dumps(x) + "\n" for x in events))
            events.clear()

    def wrap(fn, label):
        @functools.wraps(fn)
        def measured(*args, **kwargs):
            if not current.get("detail") or not getattr(local, "active", False):
                return fn(*args, **kwargs)
            stack = getattr(local, "stack", None)
            if stack is None:
                stack = local.stack = []
            start, cpu = time.perf_counter_ns(), time.thread_time_ns()
            frame = [0, 0]
            stack.append(frame)
            nvtx = current.get("nvtx", False) and torch.cuda.is_initialized()
            if nvtx:
                torch.cuda.nvtx.range_push(label)
            try:
                return fn(*args, **kwargs)
            finally:
                if nvtx:
                    torch.cuda.nvtx.range_pop()
                wall, used = time.perf_counter_ns() - start, time.thread_time_ns() - cpu
                stack.pop()
                if stack:
                    stack[-1][0] += wall
                    stack[-1][1] += used
                emit("function", name=label, start_ns=start, end_ns=start + wall,
                     wall_ns=wall, cpu_ns=used, self_ns=max(0, wall-frame[0]),
                     self_cpu_ns=max(0, used-frame[1]), depth=len(stack),
                     uids=getattr(local, "uids", []), phase=getattr(local, "phase", None))
        return measured

    # Reuse the project's explicit target registry, but NOT its global profiler.
    # Exclude per-token/layer functions and blocking receives; instrument their
    # coarse parents instead. Imported function aliases are replaced as well.
    excluded = {"scheduler_loop", "scheduler_idle", "scheduler_receive", "scheduler_host",
                "scheduler_metrics", "scheduler_ipc", "scheduler_reply"}
    replacements = {}
    installed = []
    for target in MINISGL_TARGETS:
        if target.stage in excluded or target.qualname in installed:
            continue
        module_name = target.filename_suffix.removesuffix(".py").replace("/", ".")
        try:
            owner = importlib.import_module(module_name)
            parts = target.qualname.split(".")
            for part in parts[:-1]:
                owner = getattr(owner, part)
            name = parts[-1]
            descriptor = inspect.getattr_static(owner, name)
            fn = descriptor.__func__ if isinstance(descriptor, (staticmethod, classmethod)) else descriptor
            if not inspect.isfunction(fn):
                continue
            replacement = wrap(fn, target.qualname)
            replacements[fn] = replacement
            setattr(owner, name, type(descriptor)(replacement)
                    if isinstance(descriptor, (staticmethod, classmethod)) else replacement)
            installed.append(target.qualname)
        except (ImportError, AttributeError) as exc:
            emit("target_unavailable", name=target.qualname, error=str(exc))
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("minisgl.") and module is not None:
            for name, value in list(vars(module).items()):
                if inspect.isfunction(value) and value in replacements:
                    setattr(module, name, replacements[value])
    # The registry lacks a few important coarse current-source paths.
    for module_name, names in {
        "minisgl.tokenizer.tokenize": ["TokenizeManager._chat_tokenize", "TokenizeManager._compile_delta_layout",
            "TokenizeManager._render_harmony_message_drop", "TokenizeManager._build_harmony_prompt",
            "TokenizeManager._build_harmony_provenance", "TokenizeManager._build_position_range_drop_plan"],
        "minisgl.scheduler.prefill": ["PrefillManager.schedule_next_batch", "PrefillAdder._try_allocate_occurrence", "PrefillAdder._occurrence_capacity_for_chunk"],
        "minisgl.attention.fi": ["FlashInferBackend._initialize_metadata_once", "FlashInferBackend.prepare_metadata"],
        "minisgl.attention.base": ["build_sliding_window_attention_batch", "build_context_attention_batch",
            "build_occurrence_attention_batch", "_try_build_context_full_attention_batch",
            "_try_build_context_sliding_attention_batch", "_try_build_occurrence_sliding_attention_batch",
            "compile_context_page_tables"],
        "minisgl.kernel.context_plan": ["try_build_context_full_plan", "try_build_context_sliding_plan",
            "try_build_occurrence_sliding_plan", "try_build_occurrence_capacity_index"],
    }.items():
        module = importlib.import_module(module_name)
        for label in names:
            if label in installed:
                continue
            if "." in label:
                cls, method = label.split(".")
                owner = getattr(module, cls, None)
            else:
                owner, method = module, label
            if owner is not None and hasattr(owner, method):
                fn, replacement = replace_callable(owner, method, lambda f: wrap(f, label))
                replacements[fn] = replacement
                installed.append(label)
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("minisgl.") and module is not None:
            for name, value in list(vars(module).items()):
                if inspect.isfunction(value) and value in replacements:
                    setattr(module, name, replacements[value])
    torch.cuda.Event.synchronize = wrap(torch.cuda.Event.synchronize, "CUDA.Event.synchronize")

    def gpu_range(fn, label):
        @functools.wraps(fn)
        def measured(*args, **kwargs):
            if not gpu_detail_enabled(current, getattr(local, "phase", None)):
                return fn(*args, **kwargs)
            name = label
            if label == "matmul_ogs":
                name += ".w13" if kwargs.get("fused_activation") is not None else ".w2"
            row = dict(kind="gpu_range", pid=os.getpid(), cell=current["cell"],
                       turn=current["turn"], name=name, layer=getattr(local, "layer", None),
                       uids=getattr(local, "uids", [])[:], start_ns=time.perf_counter_ns())
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                return fn(*args, **kwargs)
            finally:
                stop.record()
                row["end_ns"] = time.perf_counter_ns()
                pending_ranges.append((row, start, stop))
        return measured

    # Process-local GPU ranges complement CPU hooks. Events are read at idle,
    # not synchronized between layers, and do not affect graph capture/replay.
    from minisgl.models.gpt_oss import GptOssAttention, GptOssSparseMoeBlock, GptOssDecoderLayer
    from minisgl.layers.attention import AttentionLayer
    from minisgl.distributed import DistributedCommunicator
    from minisgl.moe import mxfp4
    for owner, name in ((GptOssAttention,"forward"), (GptOssSparseMoeBlock,"forward"),
                        (AttentionLayer,"forward"), (DistributedCommunicator,"all_reduce")):
        replace_callable(owner, name, lambda fn, label=owner.__name__+"."+name: gpu_range(fn,label))
    mxfp4._route = gpu_range(mxfp4._route, "mxfp4._route")
    original_abi = mxfp4._load_triton_kernels_abi
    def load_abi():
        abi = original_abi()
        abi.matmul_ogs = gpu_range(abi.matmul_ogs, "matmul_ogs")
        return abi
    mxfp4._load_triton_kernels_abi = load_abi
    original_layer = GptOssDecoderLayer.forward
    def layer(self, *args, **kwargs):
        previous = getattr(local, "layer", None)
        local.layer = self._layer_id
        try:
            return original_layer(self, *args, **kwargs)
        finally:
            local.layer = previous
    GptOssDecoderLayer.forward = layer

    original_chat = TokenizeManager._chat_tokenize
    def chat(self, msg):
        control()
        local.active, local.uids, local.phase = True, [msg.uid], "tokenizer"
        start = time.perf_counter_ns()
        try:
            return original_chat(self, msg)
        finally:
            emit("tokenizer", uid=msg.uid, start_ns=start, end_ns=time.perf_counter_ns())
            local.active = False
            flush()
    TokenizeManager._chat_tokenize = chat

    original_process = Scheduler._process_one_msg
    def process(self, msg):
        if isinstance(msg, UserMsg):
            control()
            local.active = False
            if current["cell"] not in cleared:
                if self.prefill_manager.pending_list or self.decode_manager.runnable:
                    raise RuntimeError("Refusing to reset KV while requests are active")
                cache = self.cache_manager
                if cache.available_size != cache.num_pages:
                    raise RuntimeError("Live KV handles prevent paired-cell isolation")
                pages = cache._allocate(cache.num_pages)
                cache.free_occurrence_pages(pages)
                cache.check_integrity()
                cleared.add(current["cell"])
                emit("cache_reset", free=len(cache.free_slots), total=cache.num_pages)
            gate.arrive((current["cell"], current["turn"]), current["concurrency"], msg.uid)
            emit("arrival", uid=msg.uid, time_ns=time.perf_counter_ns())
            local.active, local.uids, local.phase = True, [msg.uid], "receive"
        return original_process(self, msg)
    Scheduler._process_one_msg = process

    original_schedule = Scheduler._schedule_next_batch
    def schedule(self):
        if current and not gate.released and current.get("barrier", True):
            if not gate.ready():
                local.active = False
                time.sleep(0.001)
                return None
            gate.released = True
            emit("gate_release", uids=sorted(gate.arrivals),
                 start_ns=gate.first_ns, end_ns=time.perf_counter_ns())
        elif current:
            gate.released = True
        local.active = bool(self.prefill_manager.pending_list or self.decode_manager.runnable)
        local.uids = [req.uid for req in self.prefill_manager.pending_list]
        local.phase = "select"
        return original_schedule(self)
    Scheduler._schedule_next_batch = schedule

    original_forward = Engine.forward_batch
    def forward(self, batch, sampling):
        nonlocal batch_number
        local.active, local.uids, local.phase = bool(current), [r.uid for r in batch.reqs], batch.phase
        batch_number += 1
        row = dict(kind="batch", pid=os.getpid(), cell=current.get("cell"),
                   turn=current.get("turn"), batch=batch_number, phase=batch.phase,
                   uids=local.uids[:], size=batch.size,
                   extend=[r.extend_len for r in batch.reqs],
                   cached=[r.cached_len for r in batch.reqs],
                   graph=self.graph_runner.can_use_cuda_graph(batch), start_ns=time.perf_counter_ns())
        start = stop = None
        if current.get("detail") or current.get("gpu_detail"):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(self.stream)
        output = original_forward(self, batch, sampling)
        if stop is not None:
            stop.record(self.stream)
        row["end_ns"] = time.perf_counter_ns()
        pending_gpu.append((row, start, stop, output))
        return output
    Engine.forward_batch = forward

    original_idle = Scheduler.run_when_idle
    def idle(self):
        local.active = False
        original_idle(self)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        emit("process_sample", time_ns=time.perf_counter_ns(), user_s=usage.ru_utime,
             sys_s=usage.ru_stime, voluntary=usage.ru_nvcsw, involuntary=usage.ru_nivcsw,
             torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
             affinity=sorted(os.sched_getaffinity(0)),
             native_threads=len(list(Path("/proc/self/task").iterdir())))
        flush()
    Scheduler.run_when_idle = idle
    emit("installed", targets=installed, env={k: os.environ.get(k) for k in
         ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM")})
    flush()
