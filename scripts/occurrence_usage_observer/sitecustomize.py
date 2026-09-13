"""Opt-in test observer: token IDs always, page lineage only with CS_USAGE_TRACE=1.

Use this directory as the E2E runner's --observer. Never include it in production
PYTHONPATH. Page tracing synchronizes GPU reads and must not measure performance.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

if "MINISGL_R2_OBSERVER_DIR" in os.environ:
    from minisgl.core import Req
    from minisgl.message.tokenizer import DetokenizeMsg
    from minisgl.scheduler.io import SchedulerIOMixin
    from minisgl.tokenizer.tokenize import TokenizeManager

    if "CS_VALIDATION_DATE" in os.environ:
        from datetime import date
        import minisgl.tokenizer.tokenize as tokenize_module

        fixed_date = date.fromisoformat(os.environ["CS_VALIDATION_DATE"])

        class ValidationDate(date):
            @classmethod
            def today(cls):
                return fixed_date

        tokenize_module.date = ValidationDate

    root = Path(os.environ["MINISGL_R2_OBSERVER_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    original_tokenize = TokenizeManager.tokenize

    def tokenize(self, messages):
        results = original_tokenize(self, messages)
        with (root / f"inputs-{os.getpid()}.jsonl").open("a") as stream:
            for message, result in zip(messages, results, strict=True):
                tensors = {}
                for name in ("input_ids", "true_positions", "radix_match_ids", "radix_positions"):
                    tensor = getattr(result, name, None)
                    tensors[name] = None if tensor is None else hashlib.sha256(
                        tensor.numpy().tobytes()
                    ).hexdigest()
                stream.write(json.dumps({"uid": int(message.uid),
                                         "warmup": bool(message.is_warmup),
                                         "tensors": tensors}) + "\n")
        return results

    TokenizeManager.tokenize = tokenize
    tokens = {}
    original_append = Req.append_host

    def append_host(self, next_token):
        original_append(self, next_token)
        if not self.is_warmup:
            tokens.setdefault(int(self.uid), []).extend(next_token.tolist())

    Req.append_host = append_host

    original_reply = SchedulerIOMixin._reply_tokenizer_rank0

    def reply(self, replies):
        result = original_reply(self, replies)
        for message in replies:
            if isinstance(message, DetokenizeMsg) and message.finished:
                # Flush once after final server timing and reply, never per token.
                with (root / f"tokens-{message.uid}.json").open("x") as stream:
                    json.dump({str(message.uid): tokens.pop(int(message.uid), [])}, stream)
        return result

    SchedulerIOMixin._reply_tokenizer_rank0 = reply

    if os.environ.get("CS_USAGE_TRACE") == "1":
        from dataclasses import fields

        import torch
        from minisgl.attention.fi import FlashInferBackend
        import minisgl.scheduler.prefill as prefill_module
        from minisgl.scheduler.reposition_occurrence import compile_occurrence_window_reference

        original_compile = prefill_module.compile_occurrence_window

        def checked_compile(*args, **kwargs):
            result = original_compile(*args, **kwargs)
            reference = compile_occurrence_window_reference(*args, **kwargs)
            for field in fields(result):
                if not torch.equal(getattr(result, field.name), getattr(reference, field.name)):
                    raise RuntimeError(f"Occurrence window differs from baseline: {field.name}")
            return result

        prefill_module.compile_occurrence_window = checked_compile

        wanted = {int(uid) for uid in os.environ.get("CS_USAGE_UIDS", "13,14,15,24,30").split(",")}
        traces = {}
        original_forward = FlashInferBackend.forward

        def rows(names, *columns):
            return [dict(zip(names, values, strict=True)) for values in zip(*columns, strict=True)]

        def forward(self, q, k, v, layer_id, batch, *, sinks=None, sliding_window=None):
            result = original_forward(
                self, q, k, v, layer_id, batch, sinks=sinks, sliding_window=sliding_window
            )
            if (sliding_window is not None or not batch.is_prefill
                    or batch.occurrence_source_pages is None
                    or getattr(batch, "_usage_trace_recorded", False)):
                return result
            if len(batch.reqs) != 1:
                raise RuntimeError("Usage diagnostic requires one request per batch.")
            req = batch.reqs[0]
            uid = int(req.uid)
            if uid not in wanted:
                return result
            batch._usage_trace_recorded = True
            if uid not in traces:
                initial = req.initial_full_match_indices.cpu().tolist()
                positions = req.occurrence_initial_source_positions.tolist()
                if len(initial) != len(positions):
                    raise RuntimeError("Initial page/position diagnostic lengths disagree.")
                traces[uid] = {
                    "uid": uid,
                    "initial_pages": rows(("page", "raw", "position"), initial,
                                          range(len(initial)), positions),
                    "visible_until": req.full_token_visible_until.tolist(),
                    "chunks": [],
                }
            segment = batch.attn_metadata.context_segments
            if segment is None:
                raise RuntimeError("Occurrence diagnostic requires full-attention segments.")
            # Read the actual device page table consumed by FlashInfer, not the
            # production usage mask or its occurrence-ID bookkeeping.
            indices = segment.indices.cpu().tolist()
            cuq = segment.cu_seqlens_q_cpu.tolist()
            cuk = segment.cu_seqlens_k_cpu.tolist()
            pairs = batch.occurrence_position_pairs.cpu().tolist()
            traces[uid]["chunks"].append({
                "birth_writes": rows(
                    ("page", "raw", "position"), batch.out_loc.cpu().tolist(),
                    range(req.cached_len, req.device_len),
                    req.true_positions[req.cached_len:req.device_len].tolist(),
                ),
                "transforms": rows(
                    ("source", "destination", "old", "new"),
                    batch.occurrence_source_pages.cpu().tolist(),
                    batch.occurrence_destination_pages.cpu().tolist(),
                    [pair[0] for pair in pairs], [pair[1] for pair in pairs],
                ),
                "attention_segments": [
                    {"query_start": req.cached_len + cuq[i],
                     "query_end": req.cached_len + cuq[i + 1],
                     "pages": indices[cuk[i]:cuk[i + 1]]}
                    for i in range(len(cuq) - 1)
                ],
            })
            # A complete trace is recoverable even when the runner terminates
            # the server process without running Python atexit callbacks.
            path = root / f"request-{uid}-{os.getpid()}.json.gz"
            with gzip.open(path, "wt") as stream:
                json.dump(traces[uid], stream)
            return result

        FlashInferBackend.forward = forward
