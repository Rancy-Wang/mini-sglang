"""Opt-in test observer: token IDs always, page lineage only with CS_USAGE_TRACE=1.

Use this directory as the E2E runner's --observer. Never include it in production
PYTHONPATH. Page tracing synchronizes GPU reads and must not measure performance.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

if "MINISGL_R2_OBSERVER_DIR" in os.environ:
    from minisgl.core import Req
    from minisgl.scheduler.io import SchedulerIOMixin
    from minisgl.message.tokenizer import DetokenizeMsg

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
        from minisgl.attention.fi import FlashInferBackend

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
