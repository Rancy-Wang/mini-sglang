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

        async def handler(request):
            payload = await request.json()
            await asyncio.sleep(.005)
            text = 'data: {"choices":[{"delta":{"content":"a"},"finish_reason":"stop"}]}\n\n'
            text += 'data: {"usage":{"prompt_tokens":3,"completion_tokens":1},"choices":[]}\n\n'
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


if __name__ == "__main__":
    unittest.main()
