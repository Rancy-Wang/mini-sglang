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


def request_body(model, request, max_tokens):
    return api.OpenAICompletionRequest(
        model=model,
        **request,
        temperature=0,
        seed=17,
        max_tokens=max_tokens,
        ignore_eos=True,
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
    def __init__(self, model):
        self.scheduler = Scheduler(
            SchedulerConfig(
                model_path=model,
                tp_info=DistributedInfo(0, 1),
                dtype=torch.bfloat16,
                offline_mode=True,
                attention_backend="fa",
                cuda_graph_bs=[1, 2, 4],
                use_pynccl=False,
                max_running_req=8,
                max_seq_len_override=512,
                num_page_override=4096,
                max_extend_tokens=2048,
            )
        )
        tokenizer = load_tokenizer(model)
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
                        request_body(self.model, request, max_tokens),
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
        source = next((w for w in result["records"] if w["owner"] == r["owner"] and w["warmup"]), r)
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
