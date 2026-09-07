"""Resident-model differential check of the real API mask and staged flows.

Only the IPC transport is replaced. API orchestration, tokenization, scheduling,
attention, sampling, cache management and detokenization remain production code.
The pump serializes scheduler iterations; serving overlap/network are not tested.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import minisgl.core as core
import torch
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserReply,
    WarmupAckMsg,
    WarmupReply,
)
from minisgl.scheduler.config import SchedulerConfig
from minisgl.scheduler.scheduler import Scheduler
from minisgl.server import api_server as api
from minisgl.tokenizer.detokenize import DetokenizeManager
from minisgl.tokenizer.server import _build_user_msg
from minisgl.tokenizer.tokenize import TokenizeManager
from minisgl.utils import load_tokenizer


def values(tensor):
    return None if tensor is None else tensor.tolist()


def messages(index=0):
    if index == 0:
        return [
            {"role": "user", "content": "Remember the number 42."},
            {"role": "assistant", "content": "I remember 42."},
            {"role": "user", "content": "What number was mentioned?"},
        ]
    return [
        {"role": "user", "content": f"Remember {index + 53}. " + "Blue sky. " * index},
        {"role": "assistant", "content": f"The number is {index + 53}."},
        {"role": "user", "content": "Repeat the number and explain briefly."},
    ]


def request_body(model, request, max_tokens, *, ignore_eos=True):
    return api.OpenAICompletionRequest(
        model=model,
        **request,
        temperature=0,
        seed=17,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
        stream=False,
    )


def preflight(model):
    # Validate every public request shape before allocating/loading the model.
    for case in ("no_drop", "cold_mask", "partial", "high", "multiple", "rolling"):
        for turn in range(3 if case == "rolling" else 1):
            for request in inputs(case, 4, turn):
                body = request_body(model, request, 16)
                api._parse_request_drop_rule(
                    drop_rule=body.drop_rule,
                    legacy_drop_message=body.drop_message,
                    messages=[msg.model_dump() for msg in body.messages],
                    radix_drop_key_mode="delta-marker",
                )
    print("R5 request preflight PASS", flush=True)


def inputs(case, bs, turn=0):
    result = []
    for i in range(bs):
        text = messages(0 if case == "cold_mask" else i)
        drop = None if case == "no_drop" else {1: [0]}
        if case in ("multiple", "rolling"):
            for j in range(1 if case == "multiple" else turn):
                text += [
                    {"role": "assistant", "content": "I remember 42."},
                    {"role": "user", "content": f"Check again {j}. What number?"},
                ]
            drop = {j: [j - 1] for j in range(1, len(text), 2)}
        result.append({"messages": text, "drop_message": drop})
    return result


class Runner:
    def __init__(self, model, *, tokenizer=None, reference_alignment=False):
        self.reference_alignment = reference_alignment
        self.forced_tokens = None
        self.full_logits = {}
        config = SchedulerConfig(
                model_path=model,
                tp_info=DistributedInfo(0, 1),
                dtype=torch.bfloat16,
                offline_mode=True,
                attention_backend="fa",
                cuda_graph_bs=[1, 4] if reference_alignment else [1, 2, 4],
                use_pynccl=False,
                max_running_req=8,
                max_seq_len_override=49152 if reference_alignment else 512,
                num_page_override=65536 if reference_alignment else 4096,
                max_extend_tokens=49152 if reference_alignment else 2048,
            )
        tokenizer = tokenizer or load_tokenizer(model)
        # The scheduler also needs EOS/stop metadata; reuse this same tokenizer.
        import minisgl.scheduler.scheduler as scheduler_module
        original_loader = scheduler_module.load_tokenizer
        scheduler_module.load_tokenizer = lambda _: tokenizer
        try:
            self.scheduler = Scheduler(config)
        finally:
            scheduler_module.load_tokenizer = original_loader
        self.tokenizer = TokenizeManager(tokenizer)
        self.detokenizer = DetokenizeManager(tokenizer)
        self.model = model
        self.pending = []
        self.records = {}
        self.owner = {}
        self.batch = None
        self.replays = 0
        self.scheduler.send_result = self.reply
        graph = self.scheduler.engine.graph_runner
        replay = graph.replay

        def observed_replay(batch):
            self.replays += 1
            for req in batch.reqs:
                self.records[req.uid]["graph_replays"] += 1
            return replay(batch)

        graph.replay = observed_replay
        sample = self.scheduler.engine.sampler.sample

        def observed_sample(logits, args):
            if self.reference_alignment:
                sampled = sample(logits, args)
                for i, req in enumerate(self.batch.reqs):
                    state = req.reference_state
                    official = state is None or state.prefill_done or state.segment_end == len(state.full_ids)
                    record = self.records[req.uid]
                    record.setdefault("sample_rows", []).append({"official": official, "prefill": self.batch.is_prefill})
                    if official:
                        # GPU snapshots are exported once per request, never per layer/token.
                        record.setdefault("logits_gpu", []).append(logits[i].detach().clone())
                        step = len(record["logits_gpu"]) - 1
                        if self.forced_tokens is not None and step < len(self.forced_tokens):
                            sampled[i] = self.forced_tokens[step]
                return sampled
            # A small diagnostic read from the existing forward; never recompute KV.
            scores, ids = logits.topk(3, dim=-1)
            scores, ids = scores.tolist(), ids.tolist()
            for req, row, score in zip(self.batch.reqs, ids, scores, strict=True):
                self.records[req.uid]["top3"].append({"ids": row, "scores": score})
            return sample(logits, args)

        self.scheduler.engine.sampler.sample = observed_sample

    async def put(self, msg):
        assert isinstance(msg, TokenizeMsg), type(msg)
        self.owner[msg.uid] = asyncio.current_task().get_name()
        self.pending.append(msg)

    def reply(self, replies):
        for msg in replies:
            record = self.records[msg.uid]
            if isinstance(msg, DetokenizeMsg):
                record["tokens"].append(msg.next_token)
                text = self.detokenizer.detokenize([msg])[0]
                fields = {
                    key: getattr(msg, key)
                    for key in (
                        "uid",
                        "finished",
                        "finish_reason",
                        "matched_stop",
                        "cached_tokens",
                        "drop_skipped_tokens",
                        "repos_tokens",
                        "prompt_tokens",
                        "completion_tokens",
                        "server_metrics",
                    )
                }
                reply = UserReply(
                    **fields, incremental_output=text, incremental_token_ids=[msg.next_token]
                )
            else:
                assert isinstance(msg, WarmupAckMsg), msg
                reply = WarmupReply(
                    **{
                        key: getattr(msg, key)
                        for key in (
                            "uid",
                            "hit_ratio",
                            "cached_tokens",
                            "drop_skipped_tokens",
                            "repos_tokens",
                            "finished",
                        )
                    }
                )
            if msg.finished:
                record["terminal"] = asdict(reply)
            self.frontend.ack_map[msg.uid].append(reply)
            self.frontend.event_map[msg.uid].set()

    def clear(self):
        s = self.scheduler
        assert not s.prefill_manager.runnable and not s.decode_manager.runnable
        s.cache_manager.check_integrity()
        cache = s.cache_manager.prefix_cache
        s.cache_manager._free(cache.evict(cache.size_info.total_size))
        s.cache_manager.check_integrity()
        assert cache.size_info.total_size == 0
        assert not self.detokenizer.decode_map

    def stage(self, req, prefill):
        record = self.records[req.uid]
        if prefill:
            cached = req.initial_active_cached_len
            if req.use_context_mask:
                raw = req.raw_positions
                cached = int(
                    (req.full_token_visible_until[raw[:cached].long()] > raw[req.cached_len]).sum()
                )
            usage = [req.reported_cached_tokens, req.drop_skipped_tokens, req.reported_repos_tokens]
            record["stages"].append(
                {
                    "mask": req.use_context_mask,
                    "cached_len": req.cached_len,
                    "radix_cached": req.radix_cached_tokens,
                    "expected_cached": cached,
                    "usage": usage,
                    "prompt_tokens": req.prompt_tokens,
                    "ids": values(req.input_ids),
                    "positions": values(req.true_positions),
                    "raw": values(req.raw_positions),
                    "visible_until": values(req.full_token_visible_until),
                    "pages": values(
                        core.get_global_ctx().page_table[req.table_idx, : req.device_len].cpu()
                    ),
                }
            )
        elif "first_decode" not in record:
            count = len(req.input_ids) - len(record["tokens"])
            record["first_decode"] = {
                "ids": values(req.input_ids[:count]),
                "positions": values(req.true_positions[:count]),
                "raw": values(req.raw_positions[:count]),
                "pages": values(
                    core.get_global_ctx().page_table[req.table_idx, : req.cached_len].cpu()
                ),
            }

    async def generate(self, mode, requests, max_tokens=16):
        self.records, self.owner = {}, {}
        self.frontend = api.FrontendManager(
            config=SimpleNamespace(
                model_path=self.model,
                contextual_prefill_mode=mode,
                radix_drop_key_mode="delta-marker",
                tool_call_parser="auto",
                reasoning_parser="auto",
                request_timeout=120,
            ),
            send_tokenizer=self,
            recv_tokenizer=None,
            initialized=True,
        )
        previous = api.get_global_state
        api.get_global_state = lambda: self.frontend
        tasks = []
        phases = []
        try:
            tasks = [
                asyncio.create_task(
                    api.v1_completions(
                        request_body(self.model, request, max_tokens, ignore_eos=not self.reference_alignment),
                        SimpleNamespace(),
                    ),
                    name=str(i),
                )
                for i, request in enumerate(requests)
            ]
            for _ in range(2048):
                # Drain event.wait/wait_for continuations before scheduling peers.
                for _ in range(8):
                    await asyncio.sleep(0)
                pending, self.pending = self.pending, []
                for msg in pending:
                    tokenized = self.tokenizer.tokenize([msg])[0]
                    self.records[msg.uid] = {
                        "uid": msg.uid,
                        "owner": self.owner[msg.uid],
                        "warmup": msg.is_warmup,
                        "target": msg.target_msg_id,
                        "tokens": [],
                        "stages": [],
                        "top3": [],
                        "graph_replays": 0,
                        "input": {
                            "ids": values(tokenized.input_ids),
                            "positions": values(tokenized.true_positions),
                            "raw": values(tokenized.raw_positions),
                        },
                        "tokenize_invocations": tokenized.tokenize_invocations,
                        "chat_template_invocations": tokenized.chat_template_invocations,
                        "reference": tokenized.staged_reference,
                    }
                    backend = _build_user_msg(msg, tokenized)
                    self.scheduler._process_one_msg(
                        BaseBackendMsg.decoder(BaseBackendMsg.encoder(backend))
                    )
                with self.scheduler.engine_stream_ctx:
                    forward = self.scheduler._schedule_next_batch()
                    if forward is not None:
                        self.batch = forward.batch
                        phases.append(
                            {
                                "prefill": self.batch.is_prefill,
                                "uids": [r.uid for r in self.batch.reqs],
                            }
                        )
                        for req in self.batch.reqs:
                            self.stage(req, self.batch.is_prefill)
                        output = self.scheduler._forward(forward)
                        self.scheduler._process_last_data((forward, output))
                if all(task.done() for task in tasks):
                    break
            else:
                raise TimeoutError("Scheduler pump exceeded bounded iteration count")
            responses = await asyncio.gather(*tasks)
            self.scheduler.cache_manager.check_integrity()
            assert not self.frontend.ack_map and not self.frontend.event_map
            self.export_logits()
            return {
                "mode": mode,
                "responses": responses,
                "records": list(self.records.values()),
                "phases": phases,
            }
        finally:
            api.get_global_state = previous
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def export_logits(self):
        self.full_logits = {}
        for uid, record in self.records.items():
            if "logits_gpu" in record:
                rows = torch.stack(record.pop("logits_gpu")).float().cpu()
                self.full_logits[uid] = rows
                scores, ids = rows.topk(10, dim=-1)
                record["top10"] = [{"ids": i, "scores": s} for i, s in zip(ids.tolist(), scores.tolist(), strict=True)]
                record["top3"] = [{"ids": x["ids"][:3], "scores": x["scores"][:3]} for x in record["top10"]]


def validate(result, case):
    errors = []
    formal = [r for r in result["records"] if not r["warmup"]]
    for r in result["records"]:
        for stage in r["stages"]:
            c, d, repos = stage["usage"]
            if (
                c != stage["expected_cached"]
                or d != stage["radix_cached"] - c
                or repos != 0
                or min(c, d) < 0
                or c + d > stage["prompt_tokens"]
            ):
                errors.append(f"usage provenance uid={r['uid']}: {stage['usage']}")
    for r in formal:
        source = r
        response = result["responses"][int(r["owner"])]
        details = response["usage"].get("prompt_tokens_details", {})
        for field in ("cached_tokens", "drop_skipped_tokens"):
            if details.get(field, 0) != source["terminal"][field]:
                errors.append(f"API usage source uid={r['uid']} field={field}")
        if r["tokens"][0] != r["top3"][len(r["stages"]) - 1]["ids"][0]:
            errors.append(f"first prefill token was not retained uid={r['uid']}")
        if (not r.get("reference") and len(r["stages"]) != 1) or r["graph_replays"] != len(r["tokens"]) - 1:
            errors.append(f"prefill/graph counts uid={r['uid']}")
        active = {k: r["first_decode"][k] for k in ("ids", "positions", "raw")}
        if not r.get("reference") and active != r["input"]:
            errors.append(f"active state uid={r['uid']}")
        if result["mode"] == "mask":
            stage = r["stages"][0]
            if case != "rolling" and stage["mask"] != (
                case in ("cold_mask", "partial", "multiple")
            ):
                errors.append(f"unexpected mask path uid={r['uid']}")
            if case == "partial" and stage["cached_len"] <= 0:
                errors.append(f"partial cache was empty uid={r['uid']}")
            if case == "high" and stage["usage"][0] <= 0:
                errors.append(f"high cache was empty uid={r['uid']}")
    warmups = [r for r in result["records"] if r["warmup"]]
    if result["mode"] == "mask" and warmups:
        errors.append("mask submitted internal warmup")
    if warmups:
        errors.append("Unexpected frontend warmup")
    return errors


def compare(a, b):
    errors = []
    aa = sorted((r for r in a["records"] if not r["warmup"]), key=lambda r: r["owner"])
    bb = sorted((r for r in b["records"] if not r["warmup"]), key=lambda r: r["owner"])
    for x, y in zip(aa, bb, strict=True):
        for key in ("tokens",):
            if x[key] != y[key]:
                if key == "tokens":
                    first = next(
                        (i for i, (u, v) in enumerate(zip(x[key], y[key])) if u != v),
                        min(len(x[key]), len(y[key])),
                    )
                    errors.append(
                        f"tokens differ for input {x['owner']} at index {first}: "
                        f"{x[key][first : first + 1]} vs {y[key][first : first + 1]}"
                    )
                else:
                    errors.append(f"{key} differs for input {x['owner']}")
        for key in ("finish_reason", "completion_tokens"):
            if x["terminal"][key] != y["terminal"][key]:
                errors.append(f"{key} differs for input {x['owner']}")
    return errors


async def experiment(runner, emit):
    failures, singles = [], {}
    confirmation_done = False
    # First recheck the original failure in two tokens, then the complete matrix.
    matrix = [("cold_mask", 1, 2)] + [
        (case, bs, 16)
        for bs in (1, 4)
        for case in ("no_drop", "cold_mask", "partial", "high", "multiple", "rolling")
    ]
    for case, bs, length in matrix:
        pair = []
        for mode in ("mask", "staged"):
            runner.clear()
            if case == "partial":
                seed = [{"messages": item["messages"][:1]} for item in inputs(case, bs)]
                emit({"seed": case, "bs": bs, **await runner.generate(mode, seed, 2)})
            elif case == "high":
                emit({"seed": case, "bs": bs, **await runner.generate(mode, inputs(case, bs), 2)})
            turns = []
            for turn in range(3 if case == "rolling" else 1):
                result = await runner.generate(mode, inputs(case, bs, turn), length)
                errors = validate(result, case)
                turns.append(result)
                emit(
                    {
                        "case": case,
                        "bs": bs,
                        "length": length,
                        "turn": turn,
                        "validation_errors": errors,
                        **result,
                    }
                )
                failures.extend((case, bs, mode, turn, error) for error in errors)
            pair.append(turns)
        for turn, (a, b) in enumerate(zip(*pair, strict=True)):
            errors = compare(a, b)
            if length == 16:
                for result in (a, b):
                    first = next(
                        r for r in result["records"] if not r["warmup"] and r["owner"] == "0"
                    )
                    key = (case, turn, result["mode"])
                    if bs == 1:
                        singles[key] = first["tokens"]
                    elif first["tokens"] != singles[key]:
                        errors.append(f"bs1/bs4 tokens differ for {result['mode']} input 0")
            emit({"comparison": case, "bs": bs, "length": length, "turn": turn, "errors": errors})
            failures.extend((case, bs, "comparison", turn, error) for error in errors)
        if errors and not confirmation_done and case == "cold_mask":
            confirmation_done = True
            repeat = []
            for mode in ("mask", "staged"):
                runner.clear()
                result = await runner.generate(mode, inputs(case, bs), 2)
                emit({"confirmation": case, "bs": bs, **result})
                repeat.append(result)
            emit({"confirmation_comparison": case, "bs": bs, "errors": compare(*repeat)})
    runner.clear()
    emit(
        {
            "summary": "FAIL" if failures else "PASS",
            "failures": failures,
            "graph_replays": runner.replays,
        }
    )
    assert not failures, failures


@torch.inference_mode()
def run():
    output = Path(os.environ.get("MINISGL_R5_OUTPUT", "/tmp/minisgl-mask-staged-r5.jsonl"))
    preflight(os.environ["MINISGL_R3_MODEL"])
    started = time.monotonic()
    with output.open("x") as handle:

        def emit(record):
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(
                {
                    k: v
                    for k, v in record.items()
                    if k
                    in (
                        "case",
                        "mode",
                        "bs",
                        "turn",
                        "comparison",
                        "errors",
                        "validation_errors",
                        "summary",
                        "failures",
                        "graph_replays",
                    )
                },
                flush=True,
            )

        runner = Runner(os.environ["MINISGL_R3_MODEL"])
        try:
            asyncio.run(experiment(runner, emit))
        finally:
            runner.scheduler.shutdown()
            print(
                f"R5 elapsed_seconds={time.monotonic() - started:.2f}; output={output}", flush=True
            )


# R7: a reference shares model/operators, but never the mask compiler or Radix.
def intervals(xs):
    result = []
    for x in xs:
        if result and result[-1][1] == x:
            result[-1][1] = x + 1
        else:
            result.append([x, x + 1])
    return result


def reference_tokenize(manager, model, request, mode="mask", max_tokens=1536):
    body = request_body(model, request, max_tokens, ignore_eos=False)
    rule, prompt = api._parse_request_drop_rule(
        drop_rule=body.drop_rule, legacy_drop_message=body.drop_message,
        messages=[x.model_dump() for x in body.messages], radix_drop_key_mode="delta-marker",
    )
    msg = TokenizeMsg(
        uid=0, text=prompt, sampling_params=core.SamplingParams(max_tokens=max_tokens),
        target_msg_id=len(prompt), drop_rule=rule, tools=body.tools,
        tool_choice=api._normalize_tool_choice(body.tools, body.tool_choice),
        enable_thinking=body.enable_thinking, reasoning_effort=body.reasoning_effort,
        use_context_mask=mode == "mask", staged_reference=mode == "staged",
    )
    return msg, manager.tokenize([msg])[0]


def visibility_oracle(tokenized):
    """Lifetime oracle from sparse event ranges; never reads the compiled mask."""
    full = tokenized.full_input_ids
    if full is None:
        full = tokenized.input_ids
    expiry = [2**31 - 1] * len(full)
    if tokenized.drop_event_positions is not None:
        offsets = tokenized.drop_range_offsets.tolist()
        ranges = tokenized.drop_position_ranges.reshape(-1, 2).tolist()
        for i, boundary in enumerate(tokenized.drop_event_positions.tolist()):
            for start, end in ranges[offsets[i]:offsets[i + 1]]:
                assert 0 <= start < end <= boundary <= len(full)
                for k in range(start, end):
                    expiry[k] = min(expiry[k], boundary)
    return {"ids": full.tolist(), "expiry": expiry}


class VisibilityObserver:
    """Audit metadata actually consumed by FA, including inherited page provenance.

    Each group has contiguous query raws and no intervening Drop. Causal keys add
    exactly one corresponding token per query. Checking both endpoints plus that
    suffix identity proves every interior query without quadratic enumeration.
    """
    def __init__(self, runner):
        self.runner = runner
        self.ideals, self.pages, self.trace = {}, {}, []
        backend = runner.scheduler.engine.attn_backend
        prepare, forward = backend.prepare_metadata, backend.forward

        def observed_prepare(batch):
            result = prepare(batch)
            if batch.is_prefill:
                self.inspect(batch)
            return result

        def observed_forward(q, k, v, layer_id, batch, **kwargs):
            if batch.is_prefill:
                self.trace[-1].setdefault("attention_layers", []).append(layer_id)
            return forward(q, k, v, layer_id, batch, **kwargs)

        backend.prepare_metadata, backend.forward = observed_prepare, observed_forward

    def reset(self, ideals, *, preserve_pages=False):
        self.ideals, self.trace = ideals, []
        if not preserve_pages:
            self.pages = {}

    def inspect(self, batch):
        inverse, queries, requests = {}, [], {}
        page_table = core.get_global_ctx().page_table
        for req in batch.reqs:
            record = self.runner.records[req.uid]
            ideal = self.ideals[record.get("owner", "0")]
            raw = req.raw_positions[:req.device_len].tolist()
            ids = req.input_ids[:req.device_len].tolist()
            true = req.true_positions[:req.device_len].tolist()
            pages = page_table[req.table_idx, :req.device_len].tolist()
            for i, p in enumerate(pages):
                assert p not in inverse, "Requests share a physical KV page"
                inverse[p] = (req.uid, raw[i], ids[i], true[i])
            queries.extend((req.uid, k) for k in raw[req.cached_len:])
            requests[req.uid] = (req, ideal)
        meta = batch.attn_metadata
        seg = meta.context_segments or meta
        cu, lengths, tables = seg.cu_seqlens_q.tolist(), seg.cache_seqlens.tolist(), seg.page_table.tolist()
        event = {"uids": [r.uid for r in batch.reqs], "segments": [], "query_count": len(queries)}
        assert cu[-1] == len(queries)
        for j, (a, b) in enumerate(zip(cu, cu[1:])):
            uid = queries[a][0]
            assert all(u == uid for u, _ in queries[a:b])
            req, ideal = requests[uid]
            qraw = [q for _, q in queries[a:b]]
            kp = tables[j][:lengths[j]]
            mapped = [inverse[p] for p in kp]
            assert all(u == uid for u, *_ in mapped)
            kr = [x[1] for x in mapped]
            nq = b - a
            assert kr[-nq:] == qraw, "FA causal suffix does not match the queries"
            assert all(ids == ideal["ids"][raw] and true == raw for _, raw, ids, true in mapped)
            expiries = sorted(set(ideal["expiry"]))
            cuts = [0] + [i for i in range(1, nq) if qraw[i] != qraw[i - 1] + 1 or any(qraw[i - 1] < e <= qraw[i] for e in expiries)] + [nq]
            groups = []
            for lo, hi in zip(cuts, cuts[1:]):
                checks = []
                for qi in sorted({lo, hi - 1}):
                    q = qraw[qi]
                    expected = [k for k in range(q + 1) if ideal["expiry"][k] > q]
                    actual = kr[:len(kr) - nq + qi + 1]
                    assert actual == expected, (uid, q, intervals(sorted(set(expected) - set(actual))), intervals(sorted(set(actual) - set(expected))))
                    checks.append({"query": q, "visible_raw": intervals(actual)})
                groups.append({"queries": [qraw[lo], qraw[hi - 1] + 1], "count": hi - lo, "checks": checks})
            for p, (_, raw, ids, _) in zip(kp[:-nq], mapped[:-nq], strict=True):
                assert self.pages.get(p) == (raw, ids, ideal["expiry"][raw]), ("Invalid cached provenance", p, raw)
            for p, (_, raw, ids, _) in zip(kp[-nq:], mapped[-nq:], strict=True):
                self.pages[p] = (raw, ids, ideal["expiry"][raw])
            event["segments"].append({"uid": uid, "mask": req.use_context_mask,
                "keys": intervals(kr), "pages": kp, "groups": groups,
                "cached_len": req.cached_len,
                "usage": [req.reported_cached_tokens, req.drop_skipped_tokens, req.reported_repos_tokens]})
        self.trace.append(event)


class OperatorProbe:
    """Selected real prefill rows; graph decode stays captured and unchanged."""
    def __init__(self, runner):
        import minisgl.moe.fused as fused
        self.runner, self.saved = runner, []
        self.enabled, self.layer = False, -1
        self.raws, self.indices, self.data, self.shapes = [], [], {}, {}
        self.selected = set()
        backend = runner.scheduler.engine.attn_backend
        def wrap(obj, name, build):
            original = getattr(obj, name)
            self.saved.append((obj, name, original))
            setattr(obj, name, build(original))
        def prepare(original):
            def call(batch):
                result = original(batch)
                if self.enabled and batch.is_prefill:
                    raw = [int(q) for req in batch.reqs for q in req.raw_positions[req.cached_len:req.device_len]]
                    self.indices = [i for i, q in enumerate(raw) if q in self.selected]
                    self.raws = [raw[i] for i in self.indices]
                return result
            return call
        wrap(backend, "prepare_metadata", prepare)
        def attention(original):
            def call(q, k, v, layer_id, batch, **kw):
                self.capture("q_rope", q)
                self.capture("k_rope", k)
                self.capture("v", v)
                result = original(q, k, v, layer_id, batch, **kw)
                self.capture("attention_raw", result)
                return result
            return call
        wrap(backend, "forward", attention)
        def route(original):
            def call(*args, **kwargs):
                weights, ids = original(*args, **kwargs)
                self.capture("expert_ids", ids)
                self.capture("expert_weights", weights)
                return weights, ids
            return call
        wrap(fused, "fused_topk", route)
        for index, layer in enumerate(runner.scheduler.engine.model.model.layers.op_list):
            def enter(original, index=index):
                def call(x, residual=None):
                    self.layer = index
                    self.capture("layer_x", x)
                    if residual is not None:
                        self.capture("residual", residual)
                    result = original(x, residual)
                    self.capture("layer_output", result[0])
                    self.capture("residual_output", result[1])
                    return result
                return call
            wrap(layer, "forward", enter)
            for name, op in (("qkv_projection", layer.self_attn.qkv_proj),
                             ("attention_projection", layer.self_attn.o_proj),
                             ("router", layer.mlp.gate), ("mlp", layer.mlp)):
                def observe(original, name=name):
                    def call(x, *args, **kwargs):
                        self.capture(name + "_input", x)
                        result = original(x, *args, **kwargs)
                        self.capture(name, result)
                        return result
                    return call
                wrap(op, "forward", observe)

    def capture(self, name, tensor):
        if not self.enabled or not core.get_global_ctx().batch.is_prefill or not self.indices:
            return
        key = f"{self.layer:02d}_{name}"
        # Snapshot before in-place fused operators can overwrite their input.
        self.data.setdefault(key, []).append((tuple(self.raws), tensor[self.indices].detach().clone()))
        self.shapes.setdefault(key, []).append(tuple(tensor.shape))

    def reset(self, raws):
        self.selected, self.data, self.shapes = set(raws), {}, {}
        self.enabled = True

    def export(self):
        self.enabled = False
        result = {}
        for key, items in self.data.items():
            raw = [q for qs, _ in items for q in qs]
            order = torch.tensor(sorted(range(len(raw)), key=raw.__getitem__))
            result[key] = {"raw": torch.tensor(raw)[order],
                           "values": torch.cat([t for _, t in items]).cpu()[order],
                           "batch_shapes": self.shapes[key]}
        self.data = {}
        return result

    def close(self):
        for obj, name, original in reversed(self.saved):
            setattr(obj, name, original)


def compare_probes(a, b):
    rows = []
    for key in a:
        assert key in b and torch.equal(a[key]["raw"], b[key]["raw"])
        x, y = a[key]["values"].float().flatten(1), b[key]["values"].float().flatten(1)
        d = x - y
        for i, raw in enumerate(a[key]["raw"].tolist()):
            row = {"operator": key, "raw": raw, "max_abs": float(d[i].abs().max()),
                   "relative_l2": float(d[i].norm() / x[i].norm().clamp_min(1e-12)),
                   "mask_shapes": a[key]["batch_shapes"], "reference_shapes": b[key]["batch_shapes"]}
            if key.endswith("expert_ids"):
                row["expert_sets_equal"] = sorted(x[i].tolist()) == sorted(y[i].tolist())
                row["ids"] = [x[i].int().tolist(), y[i].int().tolist()]
            if key.endswith("router"):
                topk = a[key.replace("router", "expert_ids")]["values"].shape[-1]
                for label, score in (("mask", x[i]), ("reference", y[i])):
                    vals, ids = score.topk(topk + 1)
                    row[label + "_boundary_margin"] = float(vals[-2] - vals[-1])
                    row[label + "_top_experts"] = ids.tolist()
            rows.append(row)
    return rows


def token_fixture93():
    from minisgl.kernel.radix_reposition import compile_radix_reposition_layout
    from minisgl.tokenizer.tokenize import TokenizedResult
    full = torch.arange(1000, 1093, dtype=torch.int32)
    empty = torch.empty(0, dtype=torch.int32)
    events, offsets = torch.tensor([49, 79], dtype=torch.int32), torch.tensor([0, 1, 2], dtype=torch.int32)
    spans = torch.tensor([0, 25, 25, 49], dtype=torch.int32)
    layout = compile_radix_reposition_layout(full, events, offsets, spans, empty, empty)
    keep = layout.keep_mask
    raw = torch.arange(93, dtype=torch.int32)[keep]
    visible = torch.full((93,), 2**31 - 1, dtype=torch.int32)
    visible[:25], visible[25:49] = 49, 79
    return TokenizedResult(
        input_ids=full[keep], true_positions=raw, raw_positions=raw,
        radix_input_ids=layout.records[layout.token_to_key[raw.long()]], radix_match_ids=layout.records,
        prefix_keep_mask=keep[:-1].int(), prompt_tokens=93, full_input_ids=full,
        full_token_visible_until=visible, full_keep_mask=keep.int(), drop_event_positions=events,
        drop_range_offsets=offsets, drop_position_ranges=spans, drop_effective_event_count=2,
        radix_key_virtual_mask=layout.virtual_mask, radix_key_to_token=layout.key_to_token,
        radix_token_to_key=layout.token_to_key, radix_positions=layout.positions, radix_repos_info=layout.repos_info,
    )


def private_result(tokenized, cuts=None):
    """Test-only virtual cuts support the no-drop numerical control."""
    from minisgl.tokenizer.tokenize import TokenizedResult
    full = tokenized.full_input_ids if tokenized.full_input_ids is not None else tokenized.input_ids
    positions = torch.arange(len(full), dtype=torch.int32)
    return TokenizedResult(
        input_ids=full, raw_positions=positions, true_positions=positions,
        radix_input_ids=full.long(), radix_match_ids=None,
        prefix_keep_mask=torch.ones(max(len(full) - 1, 0), dtype=torch.int32),
        prompt_tokens=len(full), staged_reference=True,
        drop_event_positions=tokenized.drop_event_positions if cuts is None else torch.tensor(cuts, dtype=torch.int32),
        drop_range_offsets=tokenized.drop_range_offsets if cuts is None else torch.zeros(len(cuts) + 1, dtype=torch.int32),
        drop_position_ranges=tokenized.drop_position_ranges if cuts is None else torch.empty(0, dtype=torch.int32),
        drop_effective_event_count=tokenized.drop_effective_event_count if cuts is None else len(cuts),
        tokenize_invocations=1, chat_template_invocations=1,
    )


def preflight_backend_fixtures(manager):
    from minisgl.message.metrics import RequestMetricsState
    from minisgl.scheduler.staged_reference import StagedReferenceState
    fixture = token_fixture93()
    msg = TokenizeMsg(uid=0, text="token fixture", sampling_params=core.SamplingParams(max_tokens=2, ignore_eos=True))
    examples = [fixture, private_result(fixture)]
    for length in (46, 49, 93, 1024):
        full = torch.arange(length, dtype=torch.int32) % 1000 + 1000
        ordinary = replace(manager._ordinary_result(msg, full), tokenize_invocations=1)
        examples.extend([ordinary, private_result(ordinary, [length // 3, 2 * length // 3])])
    for result in examples:
        backend = _build_user_msg(replace(msg, staged_reference=result.staged_reference), result)
        backend = BaseBackendMsg.decoder(BaseBackendMsg.encoder(backend))
        metrics = RequestMetricsState(request_received_ns=0, prompt_tokens=result.prompt_tokens,
                                      active_prompt_tokens=len(result.input_ids),
                                      tokenize_invocations=backend.tokenize_invocations)
        metrics.observe_token(1, visible=True)
        metrics.finish(2)
        if result.staged_reference:
            state = StagedReferenceState.from_message(backend)
            assert state.next_end(49152) > 0


def backend_generate(runner, msg, tokenized):
    runner.records = {msg.uid: {"owner": "0", "tokens": [], "top3": [], "stages": [], "graph_replays": 0}}
    replies = []
    old_reply = runner.scheduler.send_result
    def collect(items):
        replies.extend(items)
        for item in items:
            if isinstance(item, DetokenizeMsg):
                runner.records[item.uid]["tokens"].append(item.next_token)
    runner.scheduler.send_result = collect
    try:
        runner.scheduler._process_one_msg(BaseBackendMsg.decoder(BaseBackendMsg.encoder(_build_user_msg(msg, tokenized))))
        for _ in range(msg.sampling_params.max_tokens + 128):
            with runner.scheduler.engine_stream_ctx:
                fd = runner.scheduler._schedule_next_batch()
                assert fd is not None
                runner.batch = fd.batch
                for req in fd.batch.reqs:
                    runner.stage(req, fd.batch.is_prefill)
                output = runner.scheduler._forward(fd)
                runner.scheduler._process_last_data((fd, output))
            if replies and replies[-1].finished:
                break
        else:
            raise TimeoutError("Backend fixture exceeded bound")
        runner.export_logits()
        runner.scheduler.cache_manager.check_integrity()
        return {"records": list(runner.records.values()), "replies": [asdict(x) for x in replies]}
    finally:
        runner.scheduler.send_result = old_reply


def logit_metrics(a, b):
    assert a.shape == b.shape and a.ndim == 2
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), "Non-finite comparison logits"
    delta = a - b
    max_abs = delta.abs().amax(dim=1)
    rms = delta.square().mean(dim=1).sqrt() / a.square().mean(dim=1).sqrt().clamp_min(1e-12)
    tv = (a.softmax(dim=1) - b.softmax(dim=1)).abs().sum(dim=1) * .5
    ai, bi = a.argmax(dim=1), b.argmax(dim=1)
    rows = torch.arange(len(a))
    am = (a[rows, ai] - a[rows, bi]).abs()
    bm = (b[rows, ai] - b[rows, bi]).abs()
    return [{"max_abs": float(x), "relative_rms": float(y), "tv": float(z),
             "argmax": [int(i), int(j)], "margins": [float(m), float(n)]}
            for x, y, z, i, j, m, n in zip(max_abs, rms, tv, ai, bi, am, bm, strict=True)]


def check_metrics(rows, limits):
    return [i for i, row in enumerate(rows) if any(row[k] > v for k, v in limits.items())
            or (row["argmax"][0] != row["argmax"][1] and max(row["margins"]) > 2 * row["max_abs"])]


def response_quality(tokenizer, tokens, request):
    from collections import Counter

    from minisgl.server.response_parser import ChatResponseParser
    raw = tokenizer.decode(tokens, skip_special_tokens=False)
    parser = ChatResponseParser(model_path="AgenticQwen", tools=request.get("tools") or [],
                                tool_call_parser="qwen", reasoning_parser="qwen3",
                                enable_thinking=request.get("enable_thinking"), separate_reasoning=True)
    # Match the production removal of terminal special tokens for parsing.
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    parsed = parser.parse_full(text)
    grams = Counter(tuple(tokens[i:i + 8]) for i in range(max(0, len(tokens) - 7)))
    return {"raw_text": raw, "parsed": asdict(parsed), "diagnostics": [asdict(x) for x in parser.diagnostics],
            "replacement_character": "\ufffd" in raw,
            "repeated_8grams": [{"ids": list(k), "count": v} for k, v in grams.items() if v >= 3]}


async def reference_experiment(runner, cases, prepared, out, dump):
    observer = VisibilityObserver(runner)
    failures, controls = [], []
    fixture = token_fixture93()
    ideal = visibility_oracle(fixture)
    # The numeric fixture deliberately fixes the screenshot's token positions;
    # the actual model tokenizer does not assign those exact lengths to its text.
    for hit, expected in ((49, (24, 25, 0)), (46, (46, 0, 0))):
        runner.clear()
        observer.reset({"0": ideal})
        seed_msg = TokenizeMsg(uid=100, text="seed", sampling_params=core.SamplingParams(max_tokens=1, ignore_eos=True))
        seed = replace(runner.tokenizer._ordinary_result(seed_msg, fixture.full_input_ids[:hit]), tokenize_invocations=1)
        seed_result = backend_generate(runner, seed_msg, seed)
        seed_trace = observer.trace
        observer.reset({"0": ideal}, preserve_pages=True)
        msg = replace(seed_msg, uid=101, use_context_mask=True, sampling_params=core.SamplingParams(max_tokens=2, ignore_eos=True))
        result = backend_generate(runner, msg, fixture)
        terminal = result["replies"][-1]
        assert tuple(terminal[k] for k in ("cached_tokens", "drop_skipped_tokens", "repos_tokens")) == expected
        rec = result["records"][0]
        assert len(rec["stages"]) == 1 and rec["first_decode"]["raw"] == list(range(49, 93))
        assert rec["graph_replays"] == 1 and rec["tokens"][0] == rec["top3"][0]["ids"][0]
        dump(f"image93_hit{hit}", {**result, "trace": observer.trace, "seed": seed_result, "seed_trace": seed_trace})
    runner.clear()
    observer.reset({"0": ideal})
    msg = TokenizeMsg(uid=102, text="reference fixture", staged_reference=True,
                      sampling_params=core.SamplingParams(max_tokens=2, ignore_eos=True))
    result = backend_generate(runner, msg, private_result(fixture))
    assert result["records"][0]["first_decode"]["raw"] == list(range(49, 93))
    dump("image93_reference", {**result, "trace": observer.trace})

    # Freeze the limits before reading any BCP generation results.
    for length in (93, 1024):
        full = torch.arange(length, dtype=torch.int32) % 1000 + 1000
        msg = TokenizeMsg(uid=200, text="no drop control", sampling_params=core.SamplingParams(max_tokens=2, ignore_eos=True))
        normal = replace(runner.tokenizer._ordinary_result(msg, full), tokenize_invocations=1)
        pair = []
        for name, tokenized in (("full", normal), ("repeat", normal), ("split", private_result(normal, [length // 3, 2 * length // 3]))):
            runner.clear()
            observer.reset({"0": visibility_oracle(normal)})
            runner.forced_tokens = None if not pair else pair[0][0]["records"][0]["tokens"]
            result = backend_generate(runner, replace(msg, staged_reference=name == "split"), tokenized)
            logits = runner.full_logits[200]
            pair.append((result, logits))
            dump(f"control{length}_{name}", {**result, "trace": observer.trace})
        controls.extend(logit_metrics(pair[0][1], pair[1][1]))
        controls.extend(logit_metrics(pair[0][1], pair[2][1]))
    runner.forced_tokens = None
    caps = {"max_abs": .1, "relative_rms": .01, "tv": .01}
    limits = {k: min(cap, max(1e-6, 3 * max(row[k] for row in controls))) for k, cap in caps.items()}
    control_failures = check_metrics(controls, caps)
    dump("frozen_limits", {"controls": controls, "hard_caps": caps, "limits": limits, "failures": control_failures})
    if control_failures:
        failures.append({"control_exceeds_hard_cap": control_failures})
    # Complete the ten frozen inputs even if numerical validation is failing;
    # semantic/JSON evidence remains useful and is reported separately.
    ordered = sorted(cases, key=lambda x: (str(x["case_id"]) not in ("806", "822", "844"), str(x["case_id"])))
    for case in ordered:
        key, request = str(case["case_id"]), case["request"]
        ideal = visibility_oracle(prepared[key][1])
        pair = []
        for mode in ("mask", "staged"):
            runner.clear()
            observer.reset({"0": ideal})
            started = time.monotonic()
            result = await runner.generate(mode, [request], 1536)
            rec = result["records"][0]
            assert len(result["records"]) == 1 and not rec["warmup"]
            assert rec["graph_replays"] == len(rec["tokens"]) - 1
            assert rec["tokens"][0] == rec["top10"][0]["ids"][0]
            if mode == "staged":
                assert all(x == 0 for s in rec["stages"] for x in s["usage"])
            else:
                assert len(rec["stages"]) == 1
            expected_active = [i for i, e in enumerate(ideal["expiry"]) if e > len(ideal["ids"]) - 1]
            assert rec["first_decode"]["raw"] == expected_active
            logits = runner.full_logits[rec["uid"]]
            quality = response_quality(runner.tokenizer.tokenizer, rec["tokens"], request)
            result.update(trace=observer.trace, quality=quality, seconds=time.monotonic() - started)
            dump(f"{key}_{mode}", result)
            torch.save(logits, out / f"{key}_{mode}_logits.pt")
            pair.append((result, logits))
            print("BCP", key, mode, "tokens", len(rec["tokens"]), "diagnostics", quality["diagnostics"], flush=True)
        a, b = (x[0]["records"][0]["tokens"] for x in pair)
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        common_rows = min(first + 1, len(pair[0][1]), len(pair[1][1]))
        rows = logit_metrics(pair[0][1][:common_rows], pair[1][1][:common_rows])
        replay = None
        if a != b:
            runner.clear()
            observer.reset({"0": ideal})
            runner.forced_tokens = a
            replay = await runner.generate("staged", [request], len(a))
            runner.forced_tokens = None
            rows = logit_metrics(pair[0][1], runner.full_logits[replay["records"][0]["uid"]])
            dump(f"{key}_same_prefix", {**replay, "trace": observer.trace, "forced_tokens": a})
        bad = check_metrics(rows, limits)
        if bad:
            failures.append({"case": key, "numeric_failures": bad})
        dump(f"{key}_comparison", {"first_divergence": first if a != b else None,
             "same_prefix_metrics": rows, "failures": bad, "limits": limits, "forced_replay": replay is not None})

    probe_cases = {str(f["case"]) for f in failures if "case" in f} & {"806", "844"}
    if probe_cases:
        probe = OperatorProbe(runner)
        try:
            for key in sorted(probe_cases):
                case = next(c for c in cases if str(c["case_id"]) == key)
                ideal = visibility_oracle(prepared[key][1])
                n = len(ideal["ids"])
                boundaries = prepared[key][1].drop_event_positions.tolist()
                selected = sorted({0, n - 1, n - 4, *[max(0, p - 1) for p in boundaries], *boundaries,
                                   *[i * (n - 1) // 8 for i in range(9)]})
                pair = []
                for mode in ("mask", "staged"):
                    runner.clear()
                    observer.reset({"0": ideal})
                    probe.reset(selected)
                    await runner.generate(mode, [case["request"]], 1)
                    snapshot = probe.export()
                    torch.save(snapshot, out / f"{key}_{mode}_operators.pt")
                    pair.append(snapshot)
                dump(f"{key}_operator_comparison", {"probe_round": 1, "rows": compare_probes(*pair)})
        finally:
            probe.close()

    for mode in ("mask", "staged"):
        for drop in (False, True):
            requests = inputs("cold_mask" if drop else "no_drop", 4)
            ideals = {str(i): visibility_oracle(reference_tokenize(runner.tokenizer, runner.model, req)[1]) for i, req in enumerate(requests)}
            runner.clear()
            observer.reset(ideals)
            result = await runner.generate(mode, requests, 4)
            assert any(len(p["uids"]) == 4 and p["prefill"] for p in result["phases"])
            assert all(r["graph_replays"] == len(r["tokens"]) - 1 for r in result["records"])
            if not drop:
                assert all(not r["reference"] and len(r["stages"]) == 1 for r in result["records"])
            dump(f"c4_{mode}_{'drop' if drop else 'ordinary'}", {**result, "trace": observer.trace})
    runner.clear()
    dump("summary", {"semantic_status": "PASS", "numeric_status": "FAIL" if failures else "PASS",
         "failures": failures, "graph_replays": runner.replays, "limits": limits})


@torch.inference_mode()
def run_reference_alignment():
    import gzip
    import hashlib
    import subprocess
    import traceback
    out = Path(os.environ["MINISGL_R7_OUTPUT"])
    out.mkdir(exist_ok=False)
    def dump(name, value):
        with gzip.open(out / f"{name}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False)
    input_path = Path(os.environ["MINISGL_R7_INPUT"])
    cases = json.loads(gzip.decompress(input_path.read_bytes()))
    assert len(cases) == 10 and len({str(c["case_id"]) for c in cases}) == 10
    model = os.environ["MINISGL_R3_MODEL"]
    started = time.monotonic()
    tokenizer = load_tokenizer(model)
    manager = TokenizeManager(tokenizer)
    preflight_backend_fixtures(manager)
    prepared = {}
    for case in cases:
        a = reference_tokenize(manager, model, case["request"], "mask")
        b = reference_tokenize(manager, model, case["request"], "staged")
        assert torch.equal(a[1].full_input_ids, b[1].input_ids)
        assert a[1].prompt_tokens + 1536 < 49152
        assert a[1].tokenize_invocations == b[1].tokenize_invocations == 1
        assert visibility_oracle(a[1]) == visibility_oracle(b[1])
        starts = {row["msg_id"]: row["raw_start"] for row in a[1].message_meta["radix_state_starts"]}
        expected_expiry = [2**31 - 1] * len(b[1].input_ids)
        rule = case["request"]["drop_rule"]
        assert rule["type"] == "message_drop" and a[1].message_meta["target_offset"] == 0
        for event, owners in rule["drop_messages"].items():
            boundary = starts[int(event) + 1]
            for owner in owners:
                for raw in range(starts[int(owner)], starts[int(owner) + 1]):
                    expected_expiry[raw] = min(expected_expiry[raw], boundary)
        assert expected_expiry == visibility_oracle(b[1])["expiry"]
        prepared[str(case["case_id"])] = a
    dump("preflight", {"input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
         "canonical": {k: {"token_ids": visibility_oracle(v[1])["ids"], "metadata": v[1].message_meta} for k, v in prepared.items()},
         "tokenizer_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(model).glob('*') if p.is_file() and (p.name.startswith('tokenizer') or p.name in ('config.json', 'generation_config.json', 'chat_template.jinja'))},
         "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
    print("R7 preflight PASS; loading one model", flush=True)
    runner = Runner(model, tokenizer=tokenizer, reference_alignment=True)
    loaded = time.monotonic()
    status = "incomplete"
    try:
        asyncio.run(reference_experiment(runner, cases, prepared, out, dump))
        status = "completed"
    except BaseException:
        dump("failure", {"traceback": traceback.format_exc()})
        raise
    finally:
        runner.scheduler.shutdown()
        dump("runtime", {"status": status, "initialization_seconds": loaded - started,
             "experiment_seconds": time.monotonic() - loaded, "model_initializations": 1,
             "tokenizer_initializations": 1, "cuda_graph_bs": [1, 4], "graph_replays": runner.replays,
             "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "dtype": "bfloat16", "attention": "fa"})


if __name__ == "__main__":
    if os.environ.get("MINISGL_R7_SUITE") == "reference_alignment":
        run_reference_alignment()
    else:
        run()
