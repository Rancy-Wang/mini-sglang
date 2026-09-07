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
from dataclasses import asdict
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
                    record = self.records[req.uid]
                    record.setdefault("sample_rows", []).append(
                        {"official": True, "prefill": self.batch.is_prefill}
                    )
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
        source = next(
            (w for w in result["records"] if w["owner"] == r["owner"] and w["warmup"]), r
        )
        response = result["responses"][int(r["owner"])]
        details = response["usage"].get("prompt_tokens_details", {})
        for field in ("cached_tokens", "drop_skipped_tokens"):
            if details.get(field, 0) != source["terminal"][field]:
                errors.append(f"API usage source uid={r['uid']} field={field}")
        if r["tokens"][0] != r["top3"][0]["ids"][0]:
            errors.append(f"first prefill token was not retained uid={r['uid']}")
        if len(r["stages"]) != 1 or r["graph_replays"] != len(r["tokens"]) - 1:
            errors.append(f"prefill/graph counts uid={r['uid']}")
        active = {k: r["first_decode"][k] for k in ("ids", "positions", "raw")}
        if active != r["input"]:
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
    if result["mode"] == "staged" and case != "no_drop" and not warmups:
        errors.append("staged did not submit real warmup")
    return errors


def compare(a, b):
    errors = []
    aa = sorted((r for r in a["records"] if not r["warmup"]), key=lambda r: r["owner"])
    bb = sorted((r for r in b["records"] if not r["warmup"]), key=lambda r: r["owner"])
    for x, y in zip(aa, bb, strict=True):
        for key in ("input", "tokens"):
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


def reference_tokenize(manager, model, request, mode="mask", max_tokens=1536):
    assert mode == "mask", "R8 validates the default production mask path."
    body = request_body(model, request, max_tokens, ignore_eos=False)
    rule, prompt = api._parse_request_drop_rule(
        drop_rule=body.drop_rule,
        legacy_drop_message=body.drop_message,
        messages=[x.model_dump() for x in body.messages],
        radix_drop_key_mode="delta-marker",
    )
    msg = TokenizeMsg(
        uid=0,
        text=prompt,
        sampling_params=core.SamplingParams(max_tokens=max_tokens),
        target_msg_id=len(prompt),
        drop_rule=rule,
        tools=body.tools,
        tool_choice=api._normalize_tool_choice(body.tools, body.tool_choice),
        enable_thinking=body.enable_thinking,
        reasoning_effort=body.reasoning_effort,
        use_context_mask=True,
    )
    return msg, manager.tokenize([msg])[0]


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


def _r8_validate_quality(tokenizer, tokens, request):
    import re

    quality = response_quality(tokenizer, tokens, request)
    assert quality["diagnostics"] == [], quality["diagnostics"]
    parsed_calls = quality["parsed"]["tool_calls"] or []
    assert parsed_calls, quality
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    assert text.count("<tool_call>") == text.count("</tool_call>") > 0
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    raw_calls = []
    for block in blocks:
        item = json.loads(block)
        raw_calls.extend(item if isinstance(item, list) else [item])
    parsed = [
        {
            "name": call["function"]["name"],
            "arguments": json.loads(call["function"]["arguments"]),
        }
        for call in parsed_calls
    ]
    expected = [
        {"name": item["name"], "arguments": item.get("arguments", {})}
        for item in raw_calls
    ]
    assert parsed == expected
    assert all(isinstance(call["arguments"], dict) for call in parsed)
    return {**quality, "raw_tool_calls": expected}


def _r8_replay_saved_logits(manager, descriptor, logits, tokens):
    class ReplayReq:
        sample_is_committed = True

        def __init__(self):
            self.sampling_params = core.SamplingParams(tool_grammar=descriptor)

    req = ReplayReq()
    first_rejected = None
    for step, token in enumerate(tokens):
        prepared = manager.prepare([req])
        assert prepared is not None
        row = logits[step : step + 1].to("cuda")
        manager.apply(row, prepared)
        if not torch.isfinite(row[0, token]):
            replacement = int(row.argmax(dim=-1).item())
            manager.accept_sampled_tokens([req], torch.tensor([replacement]), None)
            # Surface any matcher rejection immediately rather than during shutdown.
            manager.prepare([req])
            manager.apply(torch.zeros_like(row), [req])
            first_rejected = {
                "step": step,
                "baseline_token": int(token),
                "constrained_token": replacement,
            }
            break
        manager.accept_sampled_tokens([req], torch.tensor([token]), None)
    manager.discard(req)
    assert first_rejected is not None, "Invalid baseline unexpectedly passed the grammar."
    return first_rejected


@torch.inference_mode()
def run_tool_json_generation():
    import gzip
    import hashlib
    import subprocess
    import traceback

    from minisgl.engine.tool_grammar import ToolGrammarManager

    out = Path(os.environ["MINISGL_R8_OUTPUT"])
    out.mkdir(exist_ok=False)
    input_path = Path(os.environ["MINISGL_R7_INPUT"])
    baseline = Path(os.environ["MINISGL_R8_BASELINE"])
    cases = [
        case
        for case in json.loads(gzip.decompress(input_path.read_bytes()))
        if str(case["case_id"]) in {"822", "844"}
    ]
    assert {str(case["case_id"]) for case in cases} == {"822", "844"}
    cases.sort(key=lambda case: str(case["case_id"]))
    model = os.environ["MINISGL_R3_MODEL"]
    tokenizer = load_tokenizer(model)
    manager = TokenizeManager(tokenizer)
    prepared = {
        str(case["case_id"]): reference_tokenize(
            manager, model, case["request"], "mask", max_tokens=512
        )
        for case in cases
    }
    for msg, result in prepared.values():
        assert msg.sampling_params.tool_grammar is not None
        assert result.prompt_tokens + 512 < 49152

    def dump(name, value):
        with gzip.open(out / f"{name}.json.gz", "xt", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False)

    dump(
        "preflight",
        {
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "cases": {
                key: {
                    "prompt_tokens": result.prompt_tokens,
                    "tool_grammar": msg.sampling_params.tool_grammar,
                }
                for key, (msg, result) in prepared.items()
            },
        },
    )
    print("R8 preflight PASS; loading one model", flush=True)
    started = time.monotonic()
    runner = Runner(model, tokenizer=tokenizer, reference_alignment=True)
    grammar = runner.scheduler.engine.sampler.tool_grammar
    assert grammar is not None and grammar.compile_count == 0
    status = "incomplete"
    try:
        runner.clear()
        no_tools = asyncio.run(
            runner.generate(
                "mask",
                [{"messages": [{"role": "user", "content": "Answer with OK."}]}],
                2,
            )
        )
        assert grammar.compile_count == 0
        dump("no_tools_control", no_tools)

        replay_manager = ToolGrammarManager(
            tokenizer,
            runner.scheduler.engine.sampler.vocab_size,
            runner.scheduler.eos_token_ids,
        )
        replay_results = {}
        try:
            for case in cases:
                key = str(case["case_id"])
                with gzip.open(baseline / f"{key}_mask.json.gz", "rt", encoding="utf-8") as handle:
                    old = json.load(handle)
                reasons = [row["reason"] for row in old["quality"]["diagnostics"]]
                assert reasons == ["invalid_json"]
                old_tokens = old["records"][0]["tokens"]
                old_logits = torch.load(
                    baseline / f"{key}_mask_logits.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                assert len(old_logits) == len(old_tokens)
                replay_results[key] = {
                    "old_diagnostics": reasons,
                    "first_rejected": _r8_replay_saved_logits(
                        replay_manager,
                        prepared[key][0].sampling_params.tool_grammar,
                        old_logits,
                        old_tokens,
                    ),
                }
        finally:
            replay_manager.shutdown()
        dump(
            "saved_logits_replay",
            {"cases": replay_results, "compile_count": replay_manager.compile_count},
        )

        results = {}
        for case in cases:
            key = str(case["case_id"])
            results[key] = {}
            for mode in ("mask",):
                runner.clear()
                result = asyncio.run(runner.generate(mode, [case["request"]], 512))
                record = result["records"][0]
                quality = _r8_validate_quality(tokenizer, record["tokens"], case["request"])
                assert record["graph_replays"] == len(record["tokens"]) - 1
                assert result["responses"][0]["choices"][0]["message"]["tool_calls"]
                result["quality"] = quality
                results[key][mode] = result
                dump(f"{key}_{mode}", result)
                print(
                    "R8 BCP",
                    key,
                    mode,
                    "tokens",
                    len(record["tokens"]),
                    "diagnostics",
                    quality["diagnostics"],
                    flush=True,
                )
        assert grammar.compile_count > 0
        status = "PASS"
        dump(
            "summary",
            {
                "status": status,
                "grammar_compile_count": grammar.compile_count,
                "saved_logits_replay": replay_results,
                "graph_replays": runner.replays,
                "seconds": time.monotonic() - started,
            },
        )
    except BaseException:
        dump("failure", {"traceback": traceback.format_exc()})
        raise
    finally:
        runner.scheduler.shutdown()
        print(f"R8 status={status}; output={out}", flush=True)


if __name__ == "__main__":
    if os.environ.get("MINISGL_R8_SUITE") == "tool_json_generation":
        run_tool_json_generation()
    else:
        run()
