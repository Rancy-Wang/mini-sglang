#!/usr/bin/env python3
"""Isolated A800 TP2 serving experiments; smoke never starts baseline or matrix.

PLAN-CS-20260917-R3. Only child process groups created by this launcher are stopped.
The existing idle page observer provides capacity evidence and verified cold-cache
boundaries without changing the scheduler implementation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tests/benchmark'))
import test_serving as bench
# Import installs the existing observer in worker processes only when its env is set.
from run_throughput_matrix import audit, stop

MATRIX = [(c, 3 * c) for c in (1, 2, 4, 8, 16, 32)]


def validate_resources(groups):
    seen, ports = set(), set()
    for _, gpu_text, port in groups:
        gpus = gpu_text.split(',')
        if len(set(gpus)) != 2 or any(g not in ('0', '1', '2', '3') for g in gpus):
            raise ValueError('Each TP2 service must use two distinct GPUs among 0,1,2,3')
        if seen.intersection(gpus) or port in ports:
            raise ValueError('GPU sets and ports must be disjoint')
        seen.update(gpus)
        ports.add(port)


def selected_resources(args):
    if args.phase == 'experiment':
        if not 1 <= args.concurrency <= 32 or args.rounds < 1:
            raise ValueError('Experiment requires concurrency 1..32 and positive rounds')
        if args.concurrency * args.rounds > 160:
            raise ValueError('Experiment task count exceeds 160')
        gpus = args.drop_aware_gpus if args.mode == 'drop-aware' else args.ordinary_gpus
        groups = [(args.mode, gpus, args.port)]
    else:
        groups = [('drop-aware', args.drop_aware_gpus, args.port)]
        if args.phase != 'smoke':
            groups.append(('ordinary', args.ordinary_gpus, args.port + 1))
    validate_resources(groups)
    return groups


def launch(args, mode, gpus, port, root):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    usage = {i.strip(): int(v) for i, v in (line.split(',') for line in used.splitlines())}
    if any(usage[g] > 100 for g in gpus.split(',')):
        raise RuntimeError(f'Selected GPUs occupied: {usage}; no automatic process killing')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTHONPATH=str(REPO / 'python'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MINISGL_RECORD_TOKEN_TIMINGS='1',
               MINISGL_PRESERVE_HARMONY_HISTORY='1',
               MINISGL_THROUGHPUT_OBSERVER=str(root))
    for key in ('TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR', 'TVM_FFI_CACHE_DIR', 'CUDA_CACHE_PATH', 'TMPDIR'):
        path = root.parent / 'runtime' / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    for key, suffix in [('CC', 'gcc'), ('CXX', 'g++')]:
        path = Path(sys.executable).parent / ('x86_64-conda-linux-gnu-' + suffix)
        if path.exists():
            env[key] = str(path)
    if 'CXX' in env:
        env['NVCC_PREPEND_FLAGS'] = '-ccbin=' + env['CXX']
    argv = [sys.executable, str(Path(__file__).resolve()), 'worker', '--model-path', args.model,
            '--host', '127.0.0.1', '--port', str(port), '--tp-size', '2', '--dtype', 'bfloat16',
            '--disable-pynccl', '--memory-ratio', '0.9', '--max-running-requests', '32',
            '--cuda-graph-max-bs', '32', '--max-seq-len-override', '131072',
            '--max-prefill-length', '16384', '--request-timeout', '7200', '--cache-type', 'radix',
            '--page-size', '1', '--attention-backend', 'fi', '--radix-drop-key-mode', 'delta-marker',
            '--contextual-prefill-mode', 'mask', '--reposition-execution-mode', 'paged-occurrence',
            '--tool-call-parser', 'gpt-oss', '--reasoning-parser', 'gpt-oss']
    if mode == 'drop-aware':
        argv.append('--drop-aware-eviction')
    if args.pages:
        argv.extend(['--num-pages', str(args.pages)])
    with (root / 'server.log').open('x') as log:
        child = subprocess.Popen(argv, cwd=REPO, env=env, stdout=log, stderr=log,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    bench.write_json(root / 'launch.json', dict(argv=argv, pid=child.pid, gpus=gpus, head=args.head,
                     env={k: v for k, v in env.items() if k.endswith('_DIR') or k in
                          ('CUDA_VISIBLE_DEVICES', 'MINISGL_RECORD_TOKEN_TIMINGS',
                           'MINISGL_PRESERVE_HARMONY_HISTORY', 'CXX', 'CC')}))
    return child


async def ready(child, url, session, root):
    import aiohttp
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f'Server exited; inspect {root / "server.log"}')
        try:
            async with session.get(url + '/v1/models', timeout=aiohttp.ClientTimeout(total=2)) as response:
                if response.status == 200:
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        await asyncio.sleep(5)
    raise TimeoutError('Service startup exceeded 1800 seconds')


def cell_args(args, mode, port, output, c, n, *, baseline=False, baseline_path=None, case_id=None):
    values = ['run', '--host', '127.0.0.1', '--port', str(port), '--model', args.model,
              '--tokenizer', args.model, '--protocol', 'minisgl-harmony',
              '--requests-path', args.requests_path, '--output-dir', str(output),
              '--concurrency', str(c), '--num-tasks', str(n), '--seed', str(args.seed),
              '--system-tag', f'{mode}:head={args.head}:pages={args.pages or "auto"}:graph32:TP2:retain-harmony',
              '--drop' if mode == 'drop-aware' else '--no-drop']
    if baseline:
        values.append('--baseline')
    if baseline_path:
        values.extend(['--baseline-path', str(baseline_path)])
    if case_id:
        values.extend(['--case-id', str(case_id)])
    if args.phase == 'smoke':
        values.append('--smoke')
    return bench.parser().parse_args(values)


def reusable(path, identity):
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    if state.get('identity') != identity:
        raise ValueError(f'Resume identity changed: {path}')
    if state.get('state') != 'completed':
        return None
    result = Path(state['result'])
    data = json.loads(result.read_text())
    if not data['valid'] or bench.file_hash(result) != state['sha256']:
        raise ValueError(f'Invalid completed result: {result}')
    return result


async def cell(args, mode, port, output, c, n, before=None, baseline_path=None, case_id=None):
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(head=args.head, data=args.input_hash, mode=mode, c=c, n=n, seed=args.seed,
                    model=args.model, pages=args.pages, phase=args.phase, case_id=case_id,
                    baseline_hash=bench.file_hash(baseline_path) if baseline_path else None)
    status_path = output / 'status.json'
    if found := reusable(status_path, identity):
        return found
    state = dict(identity=identity, state='running', started_at=time.time())
    bench.write_json(status_path, state)
    try:
        runargs = cell_args(args, mode, port, output, c, n, baseline=before is not None,
                            baseline_path=baseline_path, case_id=case_id)
        result = await bench.run(runargs, before_task=before)
        data = json.loads(result.read_text())
        if data['overall']['actual']['all_throughput'] is None:
            raise ValueError('Missing actual forward counters')
        if any(not r['tbt_complete'] for r in data['turns'] if r['success']):
            raise ValueError('Missing complete TBT')
        state.update(state='completed', result=str(result), sha256=bench.file_hash(result), finished_at=time.time())
    except BaseException as exc:
        state.update(state='failed', error=f'{type(exc).__name__}: {exc}', finished_at=time.time())
        bench.write_json(status_path, state)
        raise
    bench.write_json(status_path, state)
    return result


async def protocol_smoke(session, url, args, root):
    """Witness real EOS termination, then force the same prompt beyond EOS."""
    base = dict(model=args.model, messages=[{'role': 'user', 'content': 'Reply with exactly OK.'}],
                max_tokens=256, temperature=0, enable_thinking=False,
                stream=True, stream_options={'include_usage': True})
    natural = await bench.request(session, url + '/v1/chat/completions', dict(base, ignore_eos=False))
    forced = await bench.request(session, url + '/v1/chat/completions', dict(base, ignore_eos=True))
    passed = (natural['finish_reason'] == 'stop' and natural['generated_tokens'] < 256
              and forced['success'] and forced['generated_tokens'] == 256 and forced['tbt_complete'])
    bench.write_json(root / 'ignore_eos.json', dict(passed=passed, natural=natural, forced=forced))
    if not passed:
        raise RuntimeError('ignore_eos probe did not demonstrate natural stop followed by forced length')
    # Deliberate server rejection must never be counted successful.
    rejected = await bench.request(session, url + '/v1/chat/completions', dict(base, max_tokens=0))
    bench.write_json(root / 'rejection.json', rejected)
    if rejected['success'] or rejected.get('http_status') != 422:
        raise RuntimeError('Malformed request not rejected as expected')


async def drop_smoke(session, url, args, root):
    """Two real generations, with synthetic tool results crossing 96K in between."""
    renderer = bench.TemplateAdapter(args.model, 'minisgl-harmony')
    tools = [{'type': 'function', 'function': {'name': 'search', 'description': 'Return text',
              'parameters': {'type': 'object', 'properties': {}}}}]
    history = [{'role': 'user', 'content': 'Read tool results and reply briefly.'}]
    state = bench.RollingState()
    full, owners = renderer.render(history, tools)
    schedule = state.extend(history, owners, full)
    first = await bench.request(session, url + '/v1/chat/completions', dict(
        model=args.model, messages=history, tools=tools, max_tokens=32, ignore_eos=True,
        stream=True, stream_options={'include_usage': True}, temperature=0))
    bench.write_json(root / 'drop_first.json', first)
    if not first['success']:
        raise RuntimeError(f"First drop smoke generation failed: {first['status']}: {first['error']}")
    history.append(first['assistant'])
    history.extend({'role': 'tool', 'name': 'search', 'tool_call_id': f'smoke_{i}',
                    'content': ' smoke' * 7100} for i in range(14))
    full, owners = renderer.render(history, tools)
    schedule = state.extend(history, owners, full)
    if not schedule['reposition'] or len(schedule['drop_message']) != 2:
        raise RuntimeError(f'Synthetic smoke did not cross 96K: {schedule}')
    second = await bench.request(session, url + '/v1/chat/completions', dict(
        model=args.model, messages=history, tools=tools, max_tokens=32, ignore_eos=True,
        stream=True, stream_options={'include_usage': True}, temperature=0,
        drop_message=schedule['drop_message'], reposition=schedule['reposition']))
    bench.write_json(root / 'drop_second.json', second)
    if not second['success']:
        raise RuntimeError(f"Second drop smoke generation failed: {second['status']}: {second['error']}")
    # A third query verifies that historical events survive verbatim while new ones append.
    previous = bench.copy.deepcopy(schedule)
    history.append(second['assistant'])
    history.append({'role': 'tool', 'name': 'search', 'tool_call_id': 'smoke_14', 'content': 'last result'})
    full, owners = renderer.render(history, tools)
    schedule = state.extend(history, owners, full)
    third = await bench.request(session, url + '/v1/chat/completions', dict(
        model=args.model, messages=history, tools=tools, max_tokens=32, ignore_eos=True,
        stream=True, stream_options={'include_usage': True}, temperature=0,
        drop_message=schedule['drop_message'], reposition=schedule['reposition']))
    bench.write_json(root / 'drop_third.json', third)
    details = (third['usage'] or {}).get('prompt_tokens_details') or {}
    # drop_skipped_tokens describes unused cache hits, not total dropped spans;
    # it may legitimately be zero when the long tool results were initially cold.
    checks = dict(
        exact_generation=second['success'] and third['success'],
        complete_tbt=second['tbt_complete'] and third['tbt_complete'],
        reposition_executed=(second['server_metrics'] or {}).get('reposition_transition_count', 0) > 0,
        long_prefix_reused=details.get('cached_tokens', 0) > previous['active_tokens'] * 0.9,
        small_incremental_prefill=0 < (third['server_metrics'] or {}).get('prefill_compute_tokens', full) < full * 0.1,
        old_drops_retained=all(schedule['drop_message'][k] == v for k, v in previous['drop_message'].items()),
        old_repositions_retained=schedule['reposition'][:len(previous['reposition'])] == previous['reposition'],
    )
    passed = all(checks.values())
    bench.write_json(root / 'drop_reposition.json', dict(passed=passed, checks=checks, synthetic=True,
                     first=first, second=second, third=third, previous=previous, final=schedule))
    if not passed:
        raise RuntimeError('Drop/reposition smoke failed; inspect saved response')


async def run(args):
    import aiohttp
    args.head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    args.input_hash = bench.file_hash(Path(args.requests_path).parent / 'manifest.json')
    root = Path(args.output_dir).resolve()
    if root == REPO or REPO in root.parents:
        raise ValueError('Outputs must be outside repository')
    root.mkdir(parents=True, exist_ok=True)
    modes = selected_resources(args)
    if args.phase == 'experiment':
        # Reject unavailable distinct tasks before reserving GPU resources.
        bench.load_cases(args.requests_path, args.concurrency * args.rounds, args.seed)
    children, resources = [], {}
    state = dict(phase=args.phase, head=args.head, state='starting', hardware='A800 TP2',
                 input_hash=args.input_hash, started_at=time.time(), pairs=[])
    bench.write_json(root / 'matrix_status.json', state)
    try:
        # Template initialization can outlast the API's idle keepalive timeout.
        # Administrative probes must not reuse a stale pooled connection. Timed
        # benchmark traffic has its own concurrency-sized pool in bench.run.
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200),
                                         connector=aiohttp.TCPConnector(force_close=True)) as session:
            for mode, gpus, port in modes:
                service_root = root / mode / ('server-' + str(time.time_ns()))
                service_root.mkdir(parents=True)
                child = launch(args, mode, gpus, port, service_root)
                children.append(child)
                resources[mode] = (port, f'http://127.0.0.1:{port}', service_root)
            await asyncio.gather(*(ready(child, resources[mode][1], session, resources[mode][2])
                                   for child, (mode, _, _) in zip(children, modes)))
            capacities = []
            for mode, _, _ in modes:
                port, url, service_root = resources[mode]
                values = [json.loads(p.read_text())['num_pages'] for p in service_root.glob('ready-*.json')]
                if len(values) != 2 or len(set(values)) != 1:
                    raise RuntimeError('Missing TP2 capacity evidence')
                capacities.append(values[0])
                await audit(session, url, service_root, 'warmup', args.model, require_compute=True)
            if len(set(capacities)) != 1:
                raise RuntimeError(f'Unequal KV capacities: {capacities}; rerun with common --pages')
            # Baseline signatures must include observed capacity, not merely "auto".
            args.pages = capacities[0]
            state['gpu_inventory'] = subprocess.check_output([
                'nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv'], text=True)
            state['model_config_sha256'] = bench.file_hash(Path(args.model) / 'config.json')
            state.update(state='running', num_pages=capacities[0])
            bench.write_json(root / 'matrix_status.json', state)

            async def clear(mode, label):
                _, url, sr = resources[mode]
                return await audit(session, url, sr, label + '-' + str(time.time_ns()), args.model, require_compute=True)

            if args.phase == 'smoke':
                mode = 'drop-aware'
                port, url, sr = resources[mode]
                await protocol_smoke(session, url, args, root)
                await clear(mode, 'before-drop')
                await drop_smoke(session, url, args, root)
                await clear(mode, 'before-trajectory')
                cases, _ = bench.load_cases(args.requests_path, 100, args.seed)
                shortest = min(cases, key=lambda x: sum(t['max_new_tokens'] for t in x['turns']))
                state['trajectory_result'] = str(await cell(args, mode, port, root / mode / 'trajectory-smoke',
                                                            1, 1, case_id=shortest['case_id']))
                await clear(mode, 'after-smoke')
            elif args.phase == 'experiment':
                mode, _, port = modes[0]
                _, url, _ = resources[mode]
                await protocol_smoke(session, url, args, root)
                if mode == 'drop-aware':
                    await clear(mode, 'before-drop-smoke')
                    await drop_smoke(session, url, args, root)
                await clear(mode, 'before-experiment')
                c, n = args.concurrency, args.concurrency * args.rounds
                state.update(concurrency=c, tasks=n, mode=mode, preflight_passed=True)
                bench.write_json(root / 'matrix_status.json', state)
                result = await cell(args, mode, port, root / mode / f'C{c}', c, n)
                if not json.loads(result.read_text())['valid']:
                    raise RuntimeError(f'Experiment did not complete all tasks: {result}')
                state['pairs'].append(dict(concurrency=c, tasks=n, results=[str(result)]))
                bench.write_json(root / 'matrix_status.json', state)
                await clear(mode, 'after-experiment')
            elif args.phase == 'baseline':
                async def baseline(mode):
                    port, _, _ = resources[mode]
                    async def before(case, instance):
                        await clear(mode, 'baseline-' + case['case_id'])
                    return await cell(args, mode, port, root / mode / 'baseline', 1, 96, before=before)
                paths = await asyncio.gather(*(baseline(mode) for mode, _, _ in modes))
                state['baseline_results'] = dict(zip((m[0] for m in modes), map(str, paths)))
            else:
                if not args.baseline_root:
                    raise ValueError('matrix requires --baseline-root from completed baseline phase')
                base_state = json.loads((Path(args.baseline_root) / 'matrix_status.json').read_text())
                if base_state['state'] != 'completed' or base_state['head'] != args.head or base_state['input_hash'] != args.input_hash:
                    raise ValueError('Baseline phase identity or completion mismatch')
                for c, n in MATRIX:
                    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() != args.head:
                        raise RuntimeError('HEAD changed during matrix')
                    await asyncio.gather(*(clear(mode, f'before-C{c}') for mode, _, _ in modes))
                    paths = await asyncio.gather(*(cell(args, mode, port, root / mode / f'C{c}', c, n,
                        baseline_path=Path(base_state['baseline_results'][mode])) for mode, _, port in modes))
                    state['pairs'].append(dict(concurrency=c, tasks=n, results=list(map(str, paths))))
                    bench.write_json(root / 'matrix_status.json', state)
                    print(json.dumps(state['pairs'][-1]), flush=True)
                    await asyncio.gather(*(clear(mode, f'after-C{c}') for mode, _, _ in modes))
            state.update(state='completed', finished_at=time.time())
    except BaseException as exc:
        state.update(state='failed', error=f'{type(exc).__name__}: {exc}', finished_at=time.time())
        raise
    finally:
        bench.write_json(root / 'matrix_status.json', state)
        for child in children:
            await asyncio.to_thread(stop, child)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        sys.argv.pop(1)
        from minisgl.server.launch import launch_server
        launch_server()
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=['smoke', 'baseline', 'matrix', 'experiment'])
    p.add_argument('--mode', choices=['drop-aware', 'ordinary'], default='drop-aware')
    p.add_argument('--concurrency', type=int, default=32)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--model', default='/mnt/public/wangruoxi/models/gpt-oss-120b')
    p.add_argument('--requests-path', default=bench.DEFAULT_INPUT)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--baseline-root')
    p.add_argument('--port', type=int, default=31080)
    p.add_argument('--drop-aware-gpus', default='0,1')
    p.add_argument('--ordinary-gpus', default='2,3')
    p.add_argument('--pages', type=int)
    p.add_argument('--seed', type=int, default=42)
    asyncio.run(run(p.parse_args()))


if __name__ == '__main__':
    main()
