"""CPU tests: real local SSE, concurrency/cutoff, windows, pinned metric oracle."""
import ast
import asyncio
import dataclasses
import os
import time
import types
import unittest
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import test_throughput as bench


def record(start=1., end=3., success=True, filler=False):
    return dict(start_time=start, end_time=end, latency=end-start, ttft=.5,
                itl=[.25, .5], text_chunks=["b", "c"], generated_text="abc",
                output_len=3, prompt_len=10, retokenized_len=3,
                success=success, strict_success=success, filler=filler, usage=None)


class MetricsTests(unittest.TestCase):
    def test_failure_and_window_accounting(self):
        records = [record(1, 3), record(2, 4, False), record(2, 5, filler=True)]
        metrics, lens = bench.calculate_metrics(records, 5)
        self.assertEqual(lens, [3, 0, 3])
        self.assertEqual(metrics["total_input"], 20)
        self.assertEqual(metrics["output_throughput"], 6/5)
        first, second = bench.summarize(records, 0, 3), bench.summarize(records, 3, 5)
        self.assertEqual(first["metrics"]["total_output"] + second["metrics"]["total_output"], 6)
        self.assertEqual(second["crossing_window_turns"], 2)
        self.assertEqual(second["filler_metrics"]["completed"], 1)

    def test_no_success(self):
        metrics, _ = bench.calculate_metrics([record(success=False)], 1)
        self.assertEqual(metrics["completed"], 0)
        self.assertIsNone(metrics["mean_e2e_latency_ms"])

    def test_pinned_sglang_oracle(self):
        path = os.environ.get("SGLANG_REFERENCE")
        if not path:
            self.skipTest("Set SGLANG_REFERENCE to pinned bench_serving.py to compare all fields")
        tree = ast.parse(Path(path).read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 and n.name in ("BenchmarkMetrics", "calculate_metrics")]
        self.assertEqual(len(nodes), 2)
        module = ast.Module(body=nodes, type_ignores=[])
        env = dict(dataclass=dataclasses.dataclass, np=np, List=List, Optional=Optional,
                   Tuple=Tuple, DatasetRow=object, RequestFuncOutput=object,
                   PreTrainedTokenizerBase=object, warnings=warnings)
        exec(compile(module, str(path), "exec"), env)
        tokenizer = types.SimpleNamespace(encode=lambda text, **kw: list(text))
        rng = np.random.default_rng(20260916)
        for _ in range(20):
            records = []
            for i in range(20):
                start = float(rng.uniform(1, 10))
                r = record(start, start + float(rng.uniform(2, 5)), i % 4 != 0)
                r["prompt_len"] = int(rng.integers(1, 10000))
                records.append(r)
            inputs = [types.SimpleNamespace(prompt_len=r["prompt_len"], text_prompt_len=r["prompt_len"], vision_prompt_len=0) for r in records]
            expected, expected_lens = env["calculate_metrics"](inputs, [types.SimpleNamespace(**r) for r in records], 20, tokenizer, "sglang-oai-chat")
            actual, lens = bench.calculate_metrics(records, 20)
            self.assertEqual(lens, expected_lens)
            self.assertEqual(set(actual), set(dataclasses.asdict(expected)))
            for key, value in dataclasses.asdict(expected).items():
                self.assertAlmostEqual(actual[key], value, places=8, msg=key)

    def test_position_threshold_not_drop_frequency(self):
        messages = [{"role": "tool"} for _ in range(6)]
        owners = [i for i in range(6) for _ in range(10)] + [-1, -1]
        schedule = bench.rolling_schedule(messages, owners, 62, threshold=40, keep=2)
        self.assertEqual(schedule["drop_message"], {"2": [0], "3": [1], "4": [2], "5": [3]})
        self.assertEqual(schedule["reposition"], [3, 5])
        self.assertEqual(schedule["position_tokens"], 22)
        self.assertEqual(schedule["reposition_checks"][0]["before"], 40)


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_client_outputs(self):
        import gzip
        import json
        import tempfile
        from unittest.mock import patch
        from aiohttp import web

        acknowledge_drop = True
        async def handler(request):
            payload = await request.json()
            await asyncio.sleep(.005)
            text = 'data: {"choices":[{"delta":{"content":"a"},"finish_reason":"stop"}]}\n\n'
            usage = {"prompt_tokens": 3, "completion_tokens": 1}
            if payload.get("drop_message") and acknowledge_drop:
                usage["prompt_tokens_details"] = {"drop_skipped_tokens": 0}
            text += 'data: ' + json.dumps({"usage": usage, "choices": []}) + '\n\n'
            text += 'data: [DONE]\n\n'
            self.assertTrue(payload["stream"])
            return web.Response(text=text, content_type="text/event-stream")
        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cases = []
                for i in range(4):
                    trajectory = [{"role": "user", "content": str(i)}]
                    filename = f"{i}.gz"
                    with gzip.open(root / filename, "wt") as stream:
                        json.dump(trajectory, stream)
                    cases.append(dict(case_id=str(i), long=i%2 == 0, file=filename,
                                      trajectory_sha256=bench.digest(trajectory), turns=[dict(
                                          turn=0, end=1, full_tokens=3, position_tokens=3,
                                          drop_message={}, reposition=[])]))
                bench.write_json(root / "manifest.json", dict(cases=cases, tools=[], tokenizer="fixture"))
                args = types.SimpleNamespace(requests_path=str(root / "manifest.json"),
                    num_requests=4, concurrency=2, max_token_len=4, tokenizer=None,
                    output=str(root / "results"), host="127.0.0.1", port=port,
                    post="/v1/chat/completions", model="fixture", api_key_env=None,
                    timeout=10, smoke_max_turns=None, smoke_long_last=False,
                    drop=False, context_limit=20)
                tokenizer = types.SimpleNamespace(encode=lambda text, **kw: list(text))
                fake = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer))
                with patch.dict("sys.modules", {"transformers": fake}):
                    await bench.run(args)
                latest = json.loads((root / "results/latest.json").read_text())
                result = json.loads(Path(latest["result"]).read_text())
                self.assertTrue(result["valid"])
                self.assertEqual(len(result["rounds"]), 2)
                self.assertEqual(len(list((root / "results").glob("*.round-*.json"))), 2)
                self.assertEqual(sum(r["metrics"]["completed"] for r in result["rounds"]), result["overall"]["metrics"]["completed"])
                # HTTP fixture tests protocol acknowledgement, not mask semantics.
                for case in cases:
                    case["turns"][0]["drop_message"] = {"0": [0]}
                bench.write_json(root / "manifest.json", dict(cases=cases, tools=[], tokenizer="fixture"))
                args.drop = True
                with patch.dict("sys.modules", {"transformers": fake}):
                    await bench.run(args)  # Zero cache savings is still valid.
                    acknowledge_drop = False
                    with self.assertRaises(SystemExit) as exc:
                        await bench.run(args)
                    self.assertEqual(exc.exception.code, 2)
        finally:
            await runner.cleanup()

    async def test_scheduler_tail_distinct_and_rounds(self):
        cases = [dict(case_id=str(i), long=i%2 == 0) for i in range(6)]
        live, events = set(), []
        async def execute(case, instance):
            self.assertNotIn(case["case_id"], live)
            live.add(case["case_id"])
            try:
                for _ in range(3):
                    await asyncio.sleep(.002 if case["case_id"] != "5" else .025)
                return "all_turns_completed"
            finally:
                live.remove(case["case_id"])
        scheduler = await bench.Scheduler(cases, 2, execute, events.append).run()
        self.assertEqual(len(scheduler.completed), 6)
        self.assertEqual(len(scheduler.round_ends), 3)
        self.assertTrue(any(x["filler"] for x in scheduler.instances))
        self.assertEqual(scheduler.cutoff, scheduler.round_ends[-1])
        self.assertFalse(live)
        self.assertTrue(all(x["start_time"] <= scheduler.cutoff for x in scheduler.instances))

    async def test_round_remainder(self):
        cases = [dict(case_id=str(i), long=False) for i in range(5)]
        async def execute(*_):
            await asyncio.sleep(.002)
            return "all_turns_completed"
        scheduler = await bench.Scheduler(cases, 2, execute).run()
        self.assertEqual(len(scheduler.round_ends), 2)
        self.assertEqual(len(scheduler.completed), 5)

    async def test_sse_usage_abort_cutoff(self):
        import aiohttp
        from aiohttp import web
        import json
        async def handler(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            mode = request.match_info["mode"]
            events = [{"choices": [{"delta": {"role": "assistant"}}]},
                      {"choices": [{"delta": {"reasoning_content": "abc"}}]},
                      {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "{}"}}]}, "finish_reason": "tool_calls"}]},
                      {"choices": [], "usage": {"completion_tokens": 7, "prompt_tokens": 100}}]
            if mode == "abort":
                events.append({"error": {"message": "aborted"}})
            for event in events:
                line = ("data: " + json.dumps(event) + "\n\n").encode()
                await response.write(line[:9])
                await response.write(line[9:])
            if mode == "wait":
                await asyncio.sleep(.2)
            else:
                await response.write(b"data: [DONE]\n\n")
            return response
        app = web.Application()
        app.router.add_post("/{mode}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://127.0.0.1:{port}"
                normal = await bench.request(session, url+"/ok", {"max_tokens": 10})
                self.assertTrue(normal["strict_success"])
                self.assertEqual(normal["output_len"], 7)
                self.assertEqual(normal["generated_text"], "abc")
                aborted = await bench.request(session, url+"/abort", {"max_tokens": 10})
                self.assertTrue(aborted["success"])
                self.assertFalse(aborted["strict_success"])
                task = asyncio.create_task(bench.request(session, url+"/wait", {"max_tokens": 10}))
                await asyncio.sleep(.05)
                task.cancel()
                cancelled = await task
                self.assertTrue(cancelled["cancelled"])
                self.assertFalse(cancelled["success"])
        finally:
            await runner.cleanup()




class ComputeAccountingTests(unittest.TestCase):
    def test_exact_acceptance_rejects_unknown_corrupt_and_incomplete_rounds(self):
        from copy import deepcopy
        from throughput_compute import require_exact_result
        rows = [record(1, 3), record(2, 5, filler=True)]
        for r in rows:
            r["events"] = [{"data": {"server_metrics": dict(
                prefill_compute_tokens=8, decode_compute_tokens=2, context_stage_count=1)}}]
        rounds = [bench.summarize(rows, 0, 3), bench.summarize(rows, 3, 5)]
        rounds[1]["cumulative"] = bench.summarize(rows, 0, 5)
        doc = dict(valid=True, args=dict(num_requests=4, concurrency=2), turns=rows,
                   overall=bench.summarize(rows, 0, 5), rounds=rounds)
        require_exact_result(doc)
        for change in ("missing_counter", "bad_round_rate", "bad_filler", "missing_round", "invalid"):
            with self.subTest(change=change):
                bad = deepcopy(doc)
                if change == "missing_counter":
                    bad["turns"][0]["events"] = []
                elif change == "bad_round_rate":
                    bad["rounds"][0]["compute_metrics"]["prefill_throughput"] = 9999
                elif change == "bad_filler":
                    bad["overall"]["filler_compute"]["decode_tokens"] += 1
                elif change == "missing_round":
                    bad["rounds"].pop()
                else:
                    bad["valid"] = False
                with self.assertRaises(ValueError):
                    require_exact_result(bad)

    def test_reuse_rope_and_first_token(self):
        from throughput_compute import turn_compute, compute_summary
        r = record()
        r.update(prompt_len=100, requested_max_tokens=10, usage={"prompt_tokens": 100,
            "completion_tokens": 3, "prompt_tokens_details": {
                "cached_tokens": 40, "drop_skipped_tokens": 20, "repos_tokens": 10}},
            reposition=[10], events=[{"data": {"server_metrics": {"generated_tokens": 3}}}])
        r["compute"] = turn_compute(r, legacy=True)
        self.assertEqual(r["compute"]["prefill_tokens"], 30)
        self.assertEqual(r["compute"]["decode_lower"], 2)
        self.assertEqual(r["compute"]["decode_upper"], 3)
        summary = compute_summary([r, record(success=False)], 2)
        self.assertEqual(summary["prefill_throughput"], 15)
        self.assertIsNone(summary["decode_throughput"])
        self.assertEqual(summary["all_throughput_lower"], 16)
        self.assertEqual(summary["all_throughput_upper"], 16.5)
        self.assertEqual(summary["completed_turns"], 1)
        self.assertIsNone(turn_compute(r)["prefill_tokens"])

    def test_mask_free_can_skip_uncached_dead_tokens(self):
        from throughput_compute import mask_extend
        # A dead suffix before the first surviving uncached query needs no forward.
        self.assertEqual(mask_extend(100, 20, [[20, 40, 40]]), (60, "compact_mask_free"))
        # A surviving query still needs those keys before their expiry: full mask.
        self.assertEqual(mask_extend(100, 20, [[20, 40, 50]]), (80, "full_mask"))
        # Warm compact cache: all dead keys are already inside the resident prefix.
        self.assertEqual(mask_extend(100, 60, [[20, 40, 50]]), (40, "compact_mask_free"))

    def test_explicit_counters_and_abort_override_usage(self):
        from throughput_compute import turn_compute, compute_summary
        r = record()
        r["events"] = [{"data": {"server_metrics": dict(prefill_compute_tokens=8,
            decode_compute_tokens=3, generated_tokens=3, context_stage_count=1)}}]
        r["compute"] = turn_compute(r)
        self.assertEqual(compute_summary([r], 2)["all_throughput"], 5.5)
        r["strict_success"] = False
        self.assertFalse(turn_compute(r)["included"])

    def test_hole_audit_rejects_legacy_reconstruction(self):
        from throughput_compute import LEGACY_HEAD, legacy_proof
        status = dict(state="completed", head=LEGACY_HEAD, input_hash="h", audit=[
            dict(passed=True, eviction=dict(drop_pages=0, hole_fills=0)) for _ in range(2)])
        evidence = dict(engine_head=LEGACY_HEAD, manifest_sha256="h",
                        profile="mask-paged-occurrence-page1-no-holes")
        self.assertTrue(legacy_proof(status, evidence, "h"))
        status["audit"][0]["eviction"]["drop_pages"] = 1
        self.assertFalse(legacy_proof(status, evidence, "h"))

    def test_forward_counts_snapshot_chunks_and_real_decode_rows(self):
        # Execute the production scheduler method with a fake engine; no CUDA is
        # needed to verify the accounting around complete_one's length mutation.
        import runpy
        repo = Path(__file__).resolve().parents[2]
        State = runpy.run_path(str(repo / "python/minisgl/message/metrics.py"))["RequestMetricsState"]
        tree = ast.parse((repo / "python/minisgl/scheduler/scheduler.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        chunk_type = type("ChunkedReq", (), {})
        env = dict(ChunkedReq=chunk_type)
        exec(compile(ast.fix_missing_locations(module), "scheduler._forward", "exec"), env)
        state = State(request_received_ns=0, prompt_tokens=50, active_prompt_tokens=40)
        req = types.SimpleNamespace(uid=1, extend_len=7, occurrence_external_storage=False,
                                    context_post_prefill_keep_mask=None)
        def forward(batch, _):
            req.extend_len = 1
            return types.SimpleNamespace(next_tokens_gpu=[1])
        obj = types.SimpleNamespace(token_pool={"in": [1]}, request_metrics={1: state},
            engine=types.SimpleNamespace(forward_batch=forward),
            decode_manager=types.SimpleNamespace(filter_reqs=lambda _: None))
        batch = types.SimpleNamespace(reqs=[req], is_prefill=True)
        class Input(tuple):
            @property
            def batch(self):
                return self[0]
        data = Input((batch, None, "in", "out"))
        env["_forward"](obj, data)
        req.extend_len = 4
        env["_forward"](obj, data)
        batch.is_prefill = False
        env["_forward"](obj, data)
        self.assertEqual(state.prefill_compute_tokens, 11)
        self.assertEqual(state.decode_compute_tokens, 1)
        state.observe_token(1, visible=True)
        api = state.finish(2).as_api_dict()
        self.assertEqual(api["prefill_compute_tokens"], 11)
        self.assertEqual(api["decode_compute_tokens"], 1)

    def test_mask_reconstruction_matches_production_reference(self):
        from throughput_compute import mask_extend
        try:
            import torch
            from minisgl.scheduler.prefill import _mask_free_context_reason_reference
        except ImportError:
            self.skipTest("Production planner comparison requires remote minisgl/torch runtime")
        prompt = 200
        spans = [[10, 30, 60], [75, 90, 110], [120, 145, 145]]
        keep = torch.ones(prompt, dtype=torch.int32)
        expiry = torch.full((prompt,), prompt+1, dtype=torch.int32)
        for lo, hi, event in spans:
            keep[lo:hi] = 0
            expiry[lo:hi] = event
        active = torch.nonzero(keep, as_tuple=False).flatten()
        req = types.SimpleNamespace(full_input_ids=torch.arange(prompt), full_keep_mask=keep,
                                    full_token_visible_until=expiry, raw_positions=active)
        for matched in range(prompt):
            cached = int(keep[:matched].sum())
            reason = _mask_free_context_reason_reference(req, active_cached_len=cached, has_sliding_window=False)
            expected = len(active)-cached if reason is None else prompt-matched
            actual, _ = mask_extend(prompt, matched, spans)
            self.assertEqual(actual, expected, msg=f"matched={matched}, reason={reason}")


    def test_recalculation_preserves_raw_and_round_additivity(self):
        import json
        import tempfile
        from throughput_compute import recalculate_file
        rows = [record(1, 3), record(2, 5, filler=True)]
        for r in rows:
            r["events"] = [{"data": {"server_metrics": dict(prefill_compute_tokens=8,
                decode_compute_tokens=2, generated_tokens=3)}}]
        overall = bench.summarize(rows, 0, 5)
        rounds = [bench.summarize(rows, 0, 3), bench.summarize(rows, 3, 5)]
        doc = dict(manifest_sha256="h", turns=rows, overall=overall, rounds=rounds)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.json"
            bench.write_json(source, doc)
            raw = source.read_bytes()
            report_path = recalculate_file(source)
            report = json.loads(report_path.read_text())
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual(report["overall"]["compute_metrics"]["prefill_tokens"], 16)
            self.assertEqual(sum(r["compute_metrics"]["prefill_tokens"] for r in report["rounds"]), 16)
            self.assertEqual(report["overall"]["sglang_logical_metrics"], overall["metrics"])
            self.assertEqual(report["overall"]["filler_compute"]["decode_tokens"], 2)
            with self.assertRaises(ValueError):
                recalculate_file(source, output_path=source)


    def test_drop_aware_insertion_holes_are_not_eviction_counters(self):
        from throughput_compute import turn_compute
        r = record()
        r.update(prompt_len=100, requested_max_tokens=10, usage={"prompt_tokens": 100,
            "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 40}}, reposition=[10])
        counts = turn_compute(r, legacy=True, allow_prefix_reconstruction=False)
        self.assertIsNone(counts["prefill_tokens"])
        self.assertIsNone(counts["decode_tokens"])
        self.assertEqual((counts["decode_lower"], counts["decode_upper"]), (2, 3))


if __name__ == "__main__":
    unittest.main()
