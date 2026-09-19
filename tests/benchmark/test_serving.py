#!/usr/bin/env python3
"""HTTP whole-trajectory serving benchmark. PLAN-CS-20260917-R3.

Inference uses HTTP only. Optional local template adapters account message boundaries;
mini-sglang's adapter is necessary for its Harmony Drop/Reposition protocol.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

# Reuse only the pinned SGLang metric formulas and SSE framing, not the old workload.
from test_throughput import REFERENCE, calculate_metrics, digest, sse_events, write_json

DEFAULT_ROOT = '/mnt/public/wangruoxi/local/throughput_bcp_full_context'
DEFAULT_INPUT = DEFAULT_ROOT + '/tasks_unique.jsonl'
DEFAULT_SOURCE = '/share/minzihan/contextualize-algo/local/analysis/bcp-formal-registry-20260916-v1/final'
THRESHOLD = 96 * 1024
SLO_LIMITS = {'ttft': [2., 3., 6.], 'tpot': [1.25, 1.5, 5.],
              'tbt': [1.25, 1.5, 5.], 'e2e': [1.25, 1.5, 5.]}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def source_key(row, source=False):
    return str(row['case_id']), int(row['aggregate_trial' if source else 'trial'])


def compile_case(trajectory, record, source):
    """Validate registry evidence; retain original messages and separate new inputs."""
    messages = trajectory['trajectory']
    calls = [c for c in record['metadata']['model_calls'] if c['side'] == 'agent']
    ends = [i for i, m in enumerate(messages) if m['role'] == 'assistant']
    if record['termination_reason'] not in ('agent_stop', 'token_limit_answer'):
        raise ValueError('incomplete_or_overflow:' + record['termination_reason'])
    if not ends or len(ends) != len(calls):
        raise ValueError('assistant_call_alignment')
    # Inspect structured protocol keys only: quoted tool text is not a Drop instruction.
    for obj in [trajectory, record, record['metadata'], *messages, *calls]:
        if any(obj.get(k) for k in ('drop_message', 'drop_rule', 'reposition')):
            raise ValueError('drop_or_reposition_present')
    names, previous, turns = {}, 0, []
    for index, (end, call) in enumerate(zip(ends, calls)):
        if call.get('context') != {'policy': 'full_context'}:
            raise ValueError('not_exclusively_full_context')
        usage = call.get('usage') or {}
        details = usage.get('prompt_tokens_details') or {}
        if any(details.get(k, 0) for k in ('drop_skipped_tokens', 'repos_tokens')):
            raise ValueError('drop_or_reposition_usage')
        length = usage.get('completion_tokens')
        if type(length) is not int or length <= 0:
            raise ValueError('missing_positive_output_length')
        inputs = copy.deepcopy(messages[previous:end])
        for msg in inputs:
            if msg['role'] in ('tool', 'function') and not msg.get('name'):
                msg['name'] = names.get(msg.get('tool_call_id'))
                if not msg['name']:
                    raise ValueError('missing_tool_name')
        turns.append(dict(turn=index, new_messages=inputs, max_new_tokens=length,
                          source_prompt_tokens=usage.get('prompt_tokens'), source_usage=usage,
                          source_assistant_message_id=end))
        for tool in messages[end].get('tool_calls') or []:
            names[tool['id']] = tool['function']['name']
        previous = end + 1
    return dict(case_id=str(trajectory['case_id']), trial=trajectory['trial'],
                trajectory=messages, turns=turns, source_record=record, source=source,
                trajectory_sha256=digest(messages))


def prepare(args):
    from minisgl.benchmark.reposition_bcp import browsecomp_plus_tools

    source = Path(args.source) / 'full_context'
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    records = {source_key(r): r for r in jsonl(source / 'records.jsonl')}
    sources = {source_key(r, True): r for r in jsonl(source / 'sources.jsonl')}
    eligible, excluded, indices = [], [], []
    with (root / 'trajectories.jsonl').open('wb') as out:
        for line, row in enumerate(jsonl(source / 'trajectories.jsonl'), 1):
            key = source_key(row)
            try:
                item = compile_case(row, records[key], sources[key])
            except (KeyError, ValueError) as exc:
                excluded.append(dict(case_id=key[0], trial=key[1], line=line, reason=str(exc)))
                continue
            item['registry_line'] = line
            raw = (json.dumps(item, ensure_ascii=False) + '\n').encode()
            indices.append(dict(case_id=key[0], trial=key[1], offset=out.tell(),
                                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            out.write(raw)
            eligible.append(dict(case_id=key[0], trial=key[1], turns=len(item['turns']),
                                 output_tokens=sum(t['max_new_tokens'] for t in item['turns'])))
    unique = {}
    for entry in sorted(indices, key=lambda x: (x['case_id'], x['trial'])):
        unique.setdefault(entry['case_id'], entry)
    for name, rows in [('tasks_unique.jsonl', list(unique.values())), ('excluded.jsonl', excluded)]:
        with (root / name).open('x') as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = dict(schema=1, source=str(source), tools=browsecomp_plus_tools(),
                    full_context_evidence='registry trajectory, agent context and usage; raw HTTP not verified',
                    source_files={name: file_hash(source / name) for name in
                                  ('trajectories.jsonl', 'records.jsonl', 'sources.jsonl')},
                    files={name: file_hash(root / name) for name in
                           ('trajectories.jsonl', 'tasks_unique.jsonl', 'excluded.jsonl')},
                    eligible=len(eligible), distinct=len(unique), excluded=len(excluded), entries=eligible)
    write_json(root / 'manifest.json', manifest)
    (root / 'README.md').write_text(
        '# Full-context serving inputs\n\n'
        'trajectories.jsonl 保留全部合格 trial 的完整源历史、调用记录、来源和新增输入切片。\n'
        'tasks_unique.jsonl 按 case_id 去重，选择最早合格 trial；offset/bytes 是 UTF-8 字节位置。\n'
        'excluded.jsonl 记录排除原因。manifest.json 包含源/输出 SHA256 与工具定义。\n'
        'full context 依据 registry 元数据，未读取无权限的原始 HTTP 文件。\n')
    print(json.dumps({k: manifest[k] for k in ('eligible', 'distinct', 'excluded')}), flush=True)


def load_cases(path, number, seed, case_ids=None):
    path = Path(path)
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    if file_hash(path) != manifest['files']['tasks_unique.jsonl']:
        raise ValueError('Task index hash mismatch')
    entries = jsonl(path)
    random.Random(seed).shuffle(entries)
    if case_ids:
        entries = [e for e in entries if e['case_id'] in case_ids]
    entries = entries[:number]
    if len(entries) != number or len({e['case_id'] for e in entries}) != number:
        raise ValueError('Not enough distinct requested tasks')
    cases = []
    with (path.parent / 'trajectories.jsonl').open('rb') as f:
        for entry in entries:
            f.seek(entry['offset'])
            raw = f.read(entry['bytes'])
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('Trajectory hash mismatch')
            row = json.loads(raw)
            if row['case_id'] != entry['case_id'] or row['trial'] != entry['trial']:
                raise ValueError('Trajectory index mismatch')
            cases.append(row)
    return cases, manifest


class RollingState:
    """Append-only events; positions derive from complete-template token ownership."""
    def __init__(self, keep=12, threshold=THRESHOLD):
        self.keep, self.threshold = keep, threshold
        self.processed = 0
        self.tools, self.drops, self.repositions, self.checks = [], {}, [], []
        self.removed = self.compacted = 0
        self.prefix_hash = None
        self.old_bounds = {}

    def extend(self, messages, owners, full_tokens):
        if self.prefix_hash is not None and digest(messages[:self.processed]) != self.prefix_hash:
            raise ValueError('Previously submitted messages changed')
        counts, ends = Counter(owners), {}
        for position, owner in enumerate(owners):
            if owner >= 0:
                ends[owner] = position + 1
        for owner, old in self.old_bounds.items():
            if old != (counts[owner], ends.get(owner)):
                raise ValueError('Template rewrote a historical message boundary')
        for i in range(self.processed, len(messages)):
            if i not in ends:
                raise ValueError(f'Message {i} has no template token ownership')
            if messages[i]['role'] in ('tool', 'function'):
                self.tools.append(i)
                if len(self.tools) > self.keep:
                    old = self.tools[-self.keep - 1]
                    self.drops[str(i)] = [old]
                    self.removed += counts[old]
            before = ends[i] - self.compacted
            if before >= self.threshold and self.removed > self.compacted:
                self.repositions.append(i)
                self.checks.append(dict(message_id=i, before=before, after=ends[i] - self.removed))
                self.compacted = self.removed
        self.processed = len(messages)
        self.prefix_hash = digest(messages)
        self.old_bounds = {i: (counts[i], ends[i]) for i in range(len(messages))}
        return dict(drop_message=copy.deepcopy(self.drops), reposition=list(self.repositions),
                    position_tokens=full_tokens - self.compacted,
                    active_tokens=full_tokens - self.removed,
                    reposition_checks=copy.deepcopy(self.checks))


class TemplateAdapter:
    def __init__(self, path, protocol):
        self.tokenizer = self.manager = None
        if path:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if protocol == 'minisgl-harmony':
            if self.tokenizer is None:
                raise ValueError('Harmony Drop requires --tokenizer')
            from minisgl.tokenizer.tokenize import TokenizeManager
            self.manager = TokenizeManager(self.tokenizer, radix_drop_key_mode='delta-marker',
                                           preserve_harmony_history=True)

    def render(self, messages, tools):
        if self.manager:
            ids, owners, _ = self.manager._render_harmony_message_drop(
                messages, enable_thinking=None, tools=tools)
            return len(ids), owners
        if self.tokenizer:
            ids = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                                                     add_generation_prompt=True)
            return len(ids), None
        return None, None

    def output_length(self, message):
        if not self.tokenizer:
            return None
        value = (message.get('reasoning_content') or '') + (message.get('content') or '')
        return len(self.tokenizer.encode(value, add_special_tokens=False))


def endpoint(host, port, post):
    if '://' not in host and host.count(':') > 1 and not host.startswith('['):
        host = '[' + host + ']'
    url = urlsplit(host if '://' in host else 'http://' + host)
    if not url.hostname or url.query or url.fragment or url.username:
        raise ValueError('Host must be an HTTP(S) base URL without credentials/query')
    if url.scheme not in ('http', 'https'):
        raise ValueError('Unsupported scheme')
    hostname = '[' + url.hostname + ']' if ':' in url.hostname else url.hostname
    selected = port if port is not None else url.port
    netloc = hostname + (f':{selected}' if selected else '')
    return urlunsplit((url.scheme, netloc, url.path.rstrip('/') + '/' + post.lstrip('/'), '', ''))


def add_delta(message, delta, tools):
    for field in ('content', 'reasoning_content'):
        value = delta.get(field)
        if field == 'reasoning_content' and value is None:
            value = delta.get('reasoning')
        if value:
            message[field] = message.get(field, '') + value
    for part in delta.get('tool_calls') or []:
        index = part.get('index', 0)
        item = tools.setdefault(index, {'type': 'function', 'id': '',
                                        'function': {'name': '', 'arguments': ''}})
        for field in ('id',):
            if part.get(field):
                item[field] += part[field]
        for field in ('name', 'arguments'):
            if (part.get('function') or {}).get(field):
                item['function'][field] += part['function'][field]


async def request(session, url, payload):
    start = time.perf_counter()
    r = dict(start_time=start, success=False, status='incomplete', error=None,
             ttft=None, latency=0., output_len=0, prompt_len=0, retokenized_len=0,
             retokenized_available=False,
             itl=[], chunk_times=[], usage=None, server_metrics=None,
             finish_reason=None, done=False, assistant={'role': 'assistant', 'content': ''})
    tools, last = {}, None
    try:
        async with session.post(url, json=payload) as response:
            r['http_status'] = response.status
            if response.status != 200:
                r.update(status='http_error', error=await response.text())
            else:
                async for data in sse_events(response.content):
                    now = time.perf_counter()
                    if data == '[DONE]':
                        r['done'] = True
                        continue
                    event = json.loads(data)
                    if event.get('error'):
                        r.update(status='server_error', error=event['error'])
                    if event.get('usage'):
                        r['usage'] = event['usage']
                    if event.get('server_metrics'):
                        r['server_metrics'] = event['server_metrics']
                    for choice in event.get('choices') or []:
                        r['finish_reason'] = choice.get('finish_reason') or r['finish_reason']
                        delta = choice.get('delta') or {}
                        if any(delta.get(k) for k in ('content', 'reasoning', 'reasoning_content', 'tool_calls')):
                            if last is None:
                                r['ttft'] = now - start
                            else:
                                r['itl'].append(now - last)
                            last = now
                            r['chunk_times'].append(now - start)
                        add_delta(r['assistant'], delta, tools)
    except asyncio.CancelledError:
        r.update(status='cutoff_cancelled', error='measurement_cutoff')
    except asyncio.TimeoutError:
        r.update(status='timeout', error='request_timeout')
    except Exception as exc:
        r.update(status='transport_error', error=f'{type(exc).__name__}: {exc}')
    r['end_time'] = time.perf_counter()
    r['latency'] = r['end_time'] - start
    if tools:
        r['assistant']['tool_calls'] = [tools[k] for k in sorted(tools)]
    usage, metrics = r['usage'] or {}, r['server_metrics'] or {}
    r['output_len'] = usage.get('completion_tokens', 0)
    r['prompt_len'] = usage.get('prompt_tokens', 0)
    r['generated_tokens'] = metrics.get('generated_tokens', usage.get('completion_tokens'))
    valid_end = r['finish_reason'] in ('length', 'stop', 'tool_calls')
    if (not r['error'] and r['done'] and valid_end and r['ttft'] is not None
            and 'prompt_tokens' in usage and 'completion_tokens' in usage):
        if r['generated_tokens'] == payload['max_tokens']:
            r.update(success=True, status='success')
        else:
            r.update(status='length_mismatch', error='Requested generation length not reached exactly')
    if not r['success'] and not r['error']:
        r['error'] = 'Missing DONE, usage, content or normal finish'
    gaps = metrics.get('token_intervals_ns')
    r['tbt_s'] = [x / 1e9 for x in gaps] if gaps is not None else None
    r['tbt_complete'] = bool(r['success'] and gaps is not None
                             and len(gaps) == r['generated_tokens'] - 1)
    if gaps is not None and (len(gaps) != (r['generated_tokens'] or 0) - 1 or any(x < 0 for x in gaps)):
        r.update(success=False, status='invalid_telemetry', error='Invalid token interval sequence', tbt_complete=False)
    r['first_decode_gap_s'] = r['tbt_s'][0] if r['tbt_s'] else None
    r['server_tpot_s'] = statistics.mean(r['tbt_s']) if r['tbt_s'] else None
    r['custom_tpot_s'] = statistics.mean(r['tbt_s'][1:]) if r['tbt_s'] and len(r['tbt_s']) > 1 else None
    r['tpot_s'] = ((r['latency'] - r['ttft']) / (r['output_len'] - 1)
                   if r['success'] and r['output_len'] > 1 else None)
    return r


class Scheduler:
    def __init__(self, cases, concurrency, execute, emit=lambda e: None, baseline=False, filler=True):
        if not 1 <= concurrency <= len(cases) <= 160:
            raise ValueError('Require 1 <= concurrency <= task count <= 160')
        if len({x['case_id'] for x in cases}) != len(cases):
            raise ValueError('Duplicate task identities')
        if baseline and concurrency != 1:
            raise ValueError('Baseline requires concurrency=1')
        self.cases, self.concurrency, self.execute, self.emit = cases, concurrency, execute, emit
        self.pending, self.active, self.completed = deque(cases), set(), []
        self.instances, self.round_ends, self.workers = [], [], []
        self.cutoff = None
        self.cursor = 0
        self.baseline = baseline
        self.filler = filler

    def choose(self):
        if self.cutoff is not None:
            return None
        if self.pending:
            return self.pending.popleft(), False
        if self.baseline or not self.filler:
            return None
        for _ in self.completed:
            case = self.completed[self.cursor % len(self.completed)]['case']
            self.cursor += 1
            if case['case_id'] not in self.active:
                return case, True
        raise RuntimeError('No distinct filler task available')

    async def worker(self, slot):
        while (selected := self.choose()) is not None:
            case, filler = selected
            key = case['case_id']
            assert key not in self.active
            self.active.add(key)
            instance = dict(instance=len(self.instances), case_id=key, trial=case['trial'],
                            filler=filler, slot=slot, start_time=time.perf_counter())
            self.instances.append(instance)
            self.emit(dict(kind='task_start', **instance))
            try:
                status = await self.execute(case, instance)
            except asyncio.CancelledError:
                status = 'cutoff_cancelled'
            except Exception as exc:
                status = 'client_error:' + str(exc)
            instance.update(status=status, end_time=time.perf_counter())
            self.active.remove(key)
            if not filler:
                self.completed.append(dict(case=case, instance=instance))
                if len(self.completed) % self.concurrency == 0:
                    self.round_ends.append(instance['end_time'])
                    self.emit(dict(kind='round_end', round=len(self.round_ends), time=instance['end_time']))
                if len(self.completed) == len(self.cases):
                    self.cutoff = instance['end_time']
                    for task in self.workers:
                        if task is not asyncio.current_task():
                            task.cancel()
            self.emit(dict(kind='task_end', **instance))
            # Failed fillers must not spin synchronously if assembly fails before HTTP.
            await asyncio.sleep(0)

    async def run(self):
        self.start = time.perf_counter()
        self.workers = [asyncio.create_task(self.worker(i)) for i in range(self.concurrency)]
        outcomes = await asyncio.gather(*self.workers, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException) and not (
                isinstance(outcome, asyncio.CancelledError) and self.cutoff is not None
            ):
                raise outcome
        return self


def stats(values):
    import numpy as np
    if not values:
        return dict(count=0, mean=None, p50=None, p90=None, p99=None, maximum=None)
    return dict(count=len(values), mean=float(np.mean(values)), maximum=max(values),
                **{f'p{p}': float(np.percentile(values, p)) for p in (50, 90, 99)})


def summary(records, start, end, split=True):
    rows = [r for r in records if start < r['end_time'] <= end]
    good = [r for r in rows if r['success']]
    duration = end - start
    logical, lengths = calculate_metrics(rows, duration)
    if any(r.get('retokenized_available') is False for r in good):
        for key in logical:
            if 'retokenized' in key:
                logical[key] = None
    actual = {}
    for stage in ('prefill', 'decode'):
        counts = [(r.get('server_metrics') or {}).get(stage + '_compute_tokens') for r in good]
        known = (bool(counts) and all(type(v) is int and v >= 0 for v in counts)
                 and all((r.get('server_metrics') or {}).get('context_stage_count', 0) <= 1 for r in good))
        actual[stage + '_tokens'] = sum(counts) if known else None
        actual[stage + '_throughput'] = sum(counts) / duration if known else None
    actual['all_throughput'] = (actual['prefill_throughput'] + actual['decode_throughput']
                                if all(actual[k + '_throughput'] is not None for k in ('prefill', 'decode')) else None)
    result = dict(duration_s=duration, actual=actual, sglang_logical=logical, output_lens=lengths,
                status_counts=dict(Counter(r['status'] for r in rows)),
                success=len(good), failed=len(rows) - len(good),
                first_pass_success=sum(not r['filler'] for r in good),
                filler_success=sum(r['filler'] for r in good),
                crossing_window_turns=sum(r['start_time'] < start for r in rows),
                tbt_seconds=stats([v for r in good for v in (r.get('tbt_s') or [])]),
                tbt_complete_requests=sum(r['tbt_complete'] for r in good),
                cached_tokens=sum(((r.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens', 0)
                                  for r in good))
    if split:
        result['first_pass'] = summary([r for r in rows if not r['filler']], start, end, False)
        result['filler'] = summary([r for r in rows if r['filler']], start, end, False)
    return result


def weighted_percentile(samples, q):
    """Inverted weighted CDF; each request contributes total weight one."""
    rows = sorted(samples)
    target, accumulated = sum(w for _, w in rows) * q / 100., 0.
    for value, weight in rows:
        accumulated += weight
        if accumulated >= target:
            return value
    return rows[-1][0] if rows else None


def slo_report(records, baseline):
    if baseline is None:
        return dict(state='unavailable', reason='No baseline supplied')
    reference = {(r['case_id'], r['trial'], r['turn']): r for r in baseline['turns']
                 if not r['filler'] and r['success']}
    ratios = {k: [] for k in SLO_LIMITS}
    weights, missing, eligible = [], [], [r for r in records if not r['filler']]
    for r in eligible:
        key = (r['case_id'], r['trial'], r['turn'])
        b = reference.get(key)
        if not r['success'] or not b or b['requested_max_tokens'] != r['requested_max_tokens']:
            missing.append(dict(key=key, reason='failure_or_missing_baseline_or_length'))
            continue
        per = {}
        for metric, field in [('ttft', 'ttft'), ('tpot', 'tpot_s'), ('e2e', 'latency')]:
            denominator, value = b.get(field), r.get(field)
            if denominator is not None and denominator > 0 and value is not None:
                per[metric] = value / denominator
                ratios[metric].append(per[metric])
            else:
                missing.append(dict(key=key, reason='undefined_' + metric))
        gaps, base_gaps = r.get('tbt_s'), b.get('tbt_s')
        if (r['tbt_complete'] and b['tbt_complete'] and gaps and base_gaps
                and len(gaps) == len(base_gaps) and all(v > 0 for v in base_gaps)):
            per['tbt'] = [v / base for v, base in zip(gaps, base_gaps)]
            ratios['tbt'].extend(per['tbt'])
            weights.extend((v, 1. / len(gaps)) for v in per['tbt'])
        else:
            missing.append(dict(key=key, reason='undefined_tbt'))
        r['slowdown'] = per
        r['baseline_prompt_delta'] = r['prompt_len'] - b['prompt_len']
    checks = {}
    for metric, limits in SLO_LIMITS.items():
        dist = stats(ratios[metric])
        if metric == 'tbt':
            dist = {f'p{p}': weighted_percentile(weights, p) if weights else None for p in (50, 90, 99)}
        checks[metric] = {f'p{p}': dict(value=dist[f'p{p}'], limit=limit,
                                      passed=dist[f'p{p}'] <= limit if dist[f'p{p}'] is not None else None)
                          for p, limit in zip((50, 90, 99), limits)}
    passed = all(c['passed'] is True for m in checks.values() for c in m.values())
    return dict(state='unavailable' if missing or not eligible else ('pass' if passed else 'fail'),
                checks=checks, missing=missing, first_pass_requests=len(eligible),
                tbt_token_weighted=stats(ratios['tbt']), tbt_main_weighting='equal_request',
                reference_hardware='A800 TP2', matching='case_id/trial/turn/token_index; reconstructed history may differ')


async def run(args, before_task=None):
    import aiohttp
    count = args.num_tasks if args.num_tasks is not None else min(5 * args.concurrency, 160)
    if not 1 <= args.concurrency <= count <= 160:
        raise ValueError('Require 1 <= concurrency <= num_tasks <= 160')
    if args.drop and args.protocol != 'minisgl-harmony':
        raise ValueError('Drop needs a protocol adapter with exact message ownership')
    if args.baseline and (args.concurrency != 1 or before_task is None):
        raise ValueError('Baseline requires C=1 and a verified cache isolation hook (use matrix launcher)')
    cases, manifest = load_cases(args.requests_path, count, args.seed, args.case_id)
    renderer = TemplateAdapter(args.tokenizer, args.protocol)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    name = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f') + f"_{'drop' if args.drop else 'no_drop'}_C{args.concurrency}"
    result_path = root / (name + '.json')
    journal_path = root / (name + '.events.jsonl')
    rows, writer_errors = [], []
    writer = ThreadPoolExecutor(max_workers=1)
    rendering = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    futures = []
    journal = journal_path.open('x')
    baseline = json.loads(Path(args.baseline_path).read_text()) if args.baseline_path else None
    signature = dict(model=args.model, protocol=args.protocol, drop=args.drop, seed=args.seed,
                     dataset=digest(manifest), tokenizer=args.tokenizer, system_tag=args.system_tag)
    if baseline and (baseline.get('signature') != signature or not baseline.get('baseline') or not baseline.get('valid')):
        raise ValueError('Baseline signature/validity mismatch')

    def emit(event):
        # Snapshot now; serialization is FIFO and outside the event loop.
        frozen = copy.deepcopy(event)
        snapshot = list(rows) if event['kind'] == 'round_end' else None
        def save():
            journal.write(json.dumps(frozen, ensure_ascii=False) + '\n')
            journal.flush()
            if event['kind'] == 'task_end':
                write_json(root / (name + '.progress.json'), frozen)
            if snapshot is not None:
                index = event['round'] - 1
                start = scheduler.start if index == 0 else scheduler.round_ends[index - 1]
                write_json(root / f'{name}.round-{event["round"]}.json', summary(snapshot, start, event['time']))
        futures.append(writer.submit(save))

    url = endpoint(args.host, args.port, args.post_path)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout),
                                    connector=aiohttp.TCPConnector(limit=args.concurrency)) as session:
        async def execute(case, instance):
            if before_task:
                await before_task(case, instance)
            history, rolling = [], RollingState()
            for turn in case['turns']:
                history.extend(copy.deepcopy(turn['new_messages']))
                assembly_start = time.perf_counter()
                full, owners = await loop.run_in_executor(rendering, renderer.render, history, manifest['tools'])
                state = rolling.extend(history, owners, full) if args.drop else {}
                payload = dict(model=args.model, messages=history, tools=manifest['tools'],
                               max_tokens=turn['max_new_tokens'], ignore_eos=True,
                               temperature=0., stream=True, stream_options={'include_usage': True})
                if args.drop:
                    payload.update(drop_message=state['drop_message'], reposition=state['reposition'])
                assembly_s = time.perf_counter() - assembly_start
                prompt_hash = digest(history)
                r = await request(session, url, payload)
                r.update(case_id=case['case_id'], trial=case['trial'], turn=turn['turn'],
                         instance=instance['instance'], filler=instance['filler'],
                         requested_max_tokens=turn['max_new_tokens'], expected_prompt_tokens=full,
                         history_sha256=prompt_hash, message_count=len(history),
                         assembly_s=assembly_s, drop_state=state,
                         source_prompt_tokens=turn['source_prompt_tokens'])
                rows.append(r)
                emit(dict(kind='turn_end', **r))
                if not r['success']:
                    return r['status']
                # Never substitute the source assistant output.
                history.append(copy.deepcopy(r['assistant']))
            return 'all_turns_completed'
        scheduler = Scheduler(cases, args.concurrency, execute, emit, baseline=args.baseline, filler=args.filler)
        try:
            await scheduler.run()
        finally:
            rendering.shutdown(wait=True)
            writer.shutdown(wait=True)
            journal.close()
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    writer_errors.append(str(exc))
    # Retokenization is outside timed work and never used to force generation length.
    for r in rows:
        length = renderer.output_length(r['assistant'])
        r.update(retokenized_len=length or 0, retokenized_available=length is not None)
    cutoff = scheduler.cutoff
    included = [r for r in rows if r['end_time'] <= cutoff]
    rounds, previous = [], scheduler.start
    for i, end in enumerate(scheduler.round_ends):
        cohort = [x['instance'] for x in scheduler.completed[i * args.concurrency:(i + 1) * args.concurrency]]
        ids = {x['instance'] for x in cohort}
        own = [r for r in included if r['instance'] in ids]
        report = dict(round=i + 1, window=summary(included, previous, end),
                      cumulative=summary(included, scheduler.start, end),
                      tasks=cohort, cohort_turn_latency_s=stats([r['latency'] for r in own if r['success']]),
                      cohort=summary(own, min(x['start_time'] for x in cohort), end),
                      cohort_task_lifetime_s=stats([x['end_time'] - x['start_time'] for x in cohort]),
                      slo=slo_report(own, baseline))
        rounds.append(report)
        write_json(root / f'{name}.round-{i + 1}.json', report)
        previous = end
    successful_tasks = sum(x['instance']['status'] == 'all_turns_completed' for x in scheduler.completed)
    slo = slo_report(included, baseline)
    if successful_tasks != count and baseline:
        slo['state'] = 'fail'
    busy = sum(max(0., min(r['end_time'], cutoff) - max(r['start_time'], scheduler.start)) for r in rows)
    result = dict(schema=1, signature=signature, reference_sglang=REFERENCE,
                  args=vars(args), url=url, baseline=args.baseline, baseline_path=args.baseline_path,
                  valid=successful_tasks == count and not writer_errors,
                  successful_tasks=successful_tasks, total_tasks=count, writer_errors=writer_errors,
                  start=scheduler.start, cutoff=cutoff, duration_s=cutoff - scheduler.start,
                  actual_http_concurrency=busy / (cutoff - scheduler.start),
                  overall=summary(included, scheduler.start, cutoff), rounds=rounds, slo=slo,
                  turns=included, excluded_at_cutoff=[r for r in rows if r['end_time'] > cutoff],
                  all_request_status_counts=dict(Counter(r['status'] for r in rows)),
                  tasks=scheduler.instances, completion_order=[x['case']['case_id'] for x in scheduler.completed],
                  journal=str(journal_path), smoke=args.smoke,
                  notes=['Successful compute tokens only; cache hits excluded; failed time retained.',
                         'Window counts assign complete turns by completion time, not GPU execution time.',
                         'TBT uses scheduler-observed sampled-token gaps; SSE chunk gaps are separate.',
                         'Custom TPOT excludes first gap, which is not pure prefill waiting.',
                         'Message reconstruction is lossy by design; source assistant outputs are not replayed.'])
    write_json(result_path, result)
    write_json(root / 'latest.json', dict(result=str(result_path), valid=result['valid']))
    print(json.dumps(dict(result=str(result_path), valid=result['valid'], actual=result['overall']['actual'])), flush=True)
    if not result['valid']:
        raise RuntimeError(f'Benchmark failed: {result_path}')
    return result_path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source', default=DEFAULT_SOURCE)
    prep.add_argument('--output-dir', default=DEFAULT_ROOT)
    runp = sub.add_parser('run')
    runp.add_argument('--concurrency', type=int, default=8)
    runp.add_argument('--num-tasks', type=int)
    runp.add_argument('--host', default='127.0.0.1')
    runp.add_argument('--port', type=int)
    runp.add_argument('--post-path', default='/v1/chat/completions')
    runp.add_argument('--model', default='gpt-oss-120b')
    runp.add_argument('--tokenizer')
    runp.add_argument('--protocol', choices=['openai', 'minisgl-harmony'], default='openai')
    runp.add_argument('--drop', action=argparse.BooleanOptionalAction, default=False)
    runp.add_argument('--requests-path', default=DEFAULT_INPUT)
    runp.add_argument('--output-dir', required=True)
    runp.add_argument('--seed', type=int, default=42)
    runp.add_argument('--case-id', action='append')
    runp.add_argument('--timeout', type=float, default=7200)
    runp.add_argument('--baseline-path')
    runp.add_argument('--system-tag', default='unspecified')
    runp.add_argument('--baseline', action='store_true')
    runp.add_argument('--filler', action=argparse.BooleanOptionalAction, default=True)
    runp.add_argument('--smoke', action='store_true', help='Label only; never truncates turns or output lengths')
    report = sub.add_parser('report')
    report.add_argument('result')
    return p


def main():
    args = parser().parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'run':
        asyncio.run(run(args))
    else:
        r = json.loads(Path(args.result).read_text())
        print(json.dumps({k: r[k] for k in ('valid', 'overall', 'slo')}, indent=2))


if __name__ == '__main__':
    main()
