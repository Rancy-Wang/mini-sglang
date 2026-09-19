from __future__ import annotations

import asyncio
import copy
import json
import unittest

import test_serving as s


def fixture():
    messages = [{'role': 'user', 'content': 'question'},
                {'role': 'assistant', 'content': 'SOURCE OUTPUT', 'tool_calls': [
                    {'id': 'old', 'function': {'name': 'search', 'arguments': '{}'}}]},
                {'role': 'tool', 'content': 'result', 'tool_call_id': 'old'},
                {'role': 'assistant', 'content': 'SOURCE FINAL'}]
    calls = [dict(side='agent', context={'policy': 'full_context'},
                  usage={'completion_tokens': n, 'prompt_tokens': 10}) for n in (3, 5)]
    row = dict(case_id='a', trial=0, trajectory=messages)
    record = dict(metadata={'model_calls': calls}, termination_reason='agent_stop')
    return row, record


class DataTests(unittest.TestCase):
    def test_full_history_and_delta_inputs(self):
        row, record = fixture()
        case = s.compile_case(row, record, {})
        self.assertEqual(case['trajectory'], row['trajectory'])
        self.assertEqual(case['turns'][1]['new_messages'], [dict(row['trajectory'][2], name='search')])
        history = case['turns'][0]['new_messages'] + [{'role': 'assistant', 'content': 'GENERATED'}]
        history += case['turns'][1]['new_messages']
        self.assertNotIn('SOURCE OUTPUT', json.dumps(history))
        self.assertEqual([x['max_new_tokens'] for x in case['turns']], [3, 5])

    def test_reject_incomplete_and_drop(self):
        row, record = fixture()
        for change in ('incomplete', 'drop', 'missing_length'):
            bad = copy.deepcopy(record)
            if change == 'incomplete':
                bad['termination_reason'] = 'incomplete'
            elif change == 'drop':
                bad['metadata']['model_calls'][0]['context']['policy'] = 'kv_drop_reposition'
            else:
                del bad['metadata']['model_calls'][0]['usage']['completion_tokens']
            with self.assertRaises(ValueError):
                s.compile_case(row, bad, {})

    def test_url(self):
        self.assertEqual(s.endpoint('http://localhost:8000', None, '/v1/chat'), 'http://localhost:8000/v1/chat')
        self.assertEqual(s.endpoint('::1', 8000, 'x'), 'http://[::1]:8000/x')


class RollingTests(unittest.TestCase):
    def test_append_events_and_current_position_threshold(self):
        state = s.RollingState(keep=2, threshold=30)
        messages = [dict(role='tool', content=str(i)) for i in range(3)]
        owners = [i for i in range(3) for _ in range(10)]
        first = state.extend(messages, owners, 32)
        self.assertEqual(first['drop_message'], {'2': [0]})
        self.assertEqual(first['reposition'], [2])
        messages.append(dict(role='tool', content='3'))
        second = state.extend(messages, owners + [3] * 10, 42)
        self.assertEqual(second['drop_message'], {'2': [0], '3': [1]})
        self.assertEqual(second['reposition'], [2, 3])
        self.assertEqual(second['position_tokens'], 22)
        messages[0]['content'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'changed'):
            state.extend(messages, owners + [3] * 10, 42)

    def test_default_12_and_no_reposition_without_holes(self):
        state = s.RollingState(threshold=1)
        messages = [dict(role='tool', content=str(i)) for i in range(12)]
        self.assertEqual(state.extend(messages, list(range(12)), 13)['drop_message'], {})
        self.assertEqual(state.repositions, [])
        messages.append(dict(role='tool', content='12'))
        out = state.extend(messages, list(range(13)), 14)
        self.assertEqual(out['drop_message'], {'12': [0]})
        self.assertEqual(out['reposition'], [12])

    def test_template_boundary_rewrite_rejected(self):
        state = s.RollingState()
        m = [dict(role='user', content='x')]
        state.extend(m, [0, 0], 2)
        with self.assertRaisesRegex(ValueError, 'boundary'):
            state.extend(m, [0], 1)


class Stream:
    def __init__(self, events):
        self.events = events
    async def __aiter__(self):
        for event in self.events:
            value = event if isinstance(event, str) else json.dumps(event)
            yield ('data: ' + value + '\n').encode()
            yield b'\n'


class Response:
    status = 200
    def __init__(self, events):
        self.content = Stream(events)
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return False


class Session:
    def __init__(self, events):
        self.events = events
    def post(self, url, json):
        return Response(self.events)


def events(n=3, error=False, done=True):
    rows = [dict(choices=[dict(delta={'reasoning_content': 'R'})]),
            dict(choices=[dict(delta={'content': 'A', 'tool_calls': [dict(index=0, id='id',
                 function=dict(name='search', arguments='{'))]})]),
            dict(choices=[dict(delta={'tool_calls': [dict(index=0, function=dict(arguments='}'))]})]),
            dict(choices=[dict(delta={}, finish_reason='length')],
                 server_metrics=dict(generated_tokens=n, token_intervals_ns=[10000000] * (n - 1),
                                     prefill_compute_tokens=10, decode_compute_tokens=n - 1)),
            dict(usage=dict(completion_tokens=n, prompt_tokens=30))]
    if error:
        rows.append({'error': 'aborted'})
    if done:
        rows.append('[DONE]')
    return rows


class RequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_output_and_fragmented_tools(self):
        r = await s.request(Session(events()), 'http://test', {'max_tokens': 3})
        self.assertTrue(r['success'])
        self.assertEqual(r['assistant']['tool_calls'][0]['function']['arguments'], '{}')
        self.assertEqual(r['assistant']['reasoning_content'], 'R')
        self.assertEqual(r['tbt_s'], [0.01, 0.01])
        self.assertEqual(r['custom_tpot_s'], 0.01)
        self.assertTrue(r['tbt_complete'])

    async def test_abort_early_eof_and_length_mismatch_fail(self):
        for rows, maximum in [(events(error=True), 3), (events(done=False), 3), (events(), 4)]:
            r = await s.request(Session(rows), 'http://test', {'max_tokens': maximum})
            self.assertFalse(r['success'])

    async def test_one_token_has_empty_complete_tbt_and_no_tpot(self):
        r = await s.request(Session(events(1)), 'http://test', {'max_tokens': 1})
        self.assertEqual(r['tbt_s'], [])
        self.assertTrue(r['tbt_complete'])
        self.assertIsNone(r['tpot_s'])
        self.assertIsNone(r['custom_tpot_s'])


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_distinct_fillers_and_cutoff(self):
        cases = [dict(case_id=str(i), trial=0) for i in range(3)]
        active, finished, fillers = set(), [], []
        async def execute(case, instance):
            key = case['case_id']
            self.assertNotIn(key, active)
            active.add(key)
            try:
                for _ in range(2):
                    await asyncio.sleep(.02 if key == '1' else .001)
                finished.append((key, instance['filler']))
                if instance['filler']:
                    fillers.append(key)
                return 'all_turns_completed'
            finally:
                active.remove(key)
        scheduler = await s.Scheduler(cases, 2, execute).run()
        self.assertEqual(len(scheduler.completed), 3)
        self.assertTrue(fillers)
        self.assertEqual(len(scheduler.round_ends), 1)
        self.assertEqual(len([x for x in finished if not x[1]]), 3)
        self.assertFalse(active)
        self.assertIsNotNone(scheduler.cutoff)

    async def test_no_filler_finishes_each_trajectory_before_reusing_slot(self):
        for concurrency in (1, 2, 4, 8):
            cases = [dict(case_id=str(i), trial=0) for i in range(3 * concurrency)]
            slots, turns, ended = {}, {}, set()
            async def execute(case, instance):
                key, slot = case['case_id'], instance['slot']
                self.assertNotIn(slot, slots)
                self.assertNotIn(key, turns)
                self.assertFalse(instance['filler'])
                slots[slot] = key
                turns[key] = []
                for turn in range(3):
                    turns[key].append(turn)
                    await asyncio.sleep(.002 if key == '0' else 0)
                    self.assertEqual(slots[slot], key)
                ended.add(key)
                del slots[slot]
                return 'http_error' if key == '1' else 'all_turns_completed'
            scheduler = await s.Scheduler(cases, concurrency, execute, filler=False).run()
            self.assertEqual(len(scheduler.instances), 3 * concurrency)
            self.assertEqual(ended, {case['case_id'] for case in cases})
            self.assertTrue(all(value == [0, 1, 2] for value in turns.values()))
            self.assertEqual(len(scheduler.round_ends), 3)
            self.assertFalse(slots)
            self.assertTrue(all(x['status'] != 'cutoff_cancelled' for x in scheduler.instances))
            self.assertEqual(sum(x['status'] == 'http_error' for x in scheduler.instances), 1)

    async def test_failure_is_terminal_and_duplicate_ids_rejected(self):
        cases = [dict(case_id='a', trial=0)]
        async def execute(*args):
            return 'http_error'
        scheduler = await s.Scheduler(cases, 1, execute).run()
        self.assertEqual(scheduler.completed[0]['instance']['status'], 'http_error')
        with self.assertRaises(ValueError):
            s.Scheduler(cases * 2, 2, execute)


class MetricTests(unittest.TestCase):
    def row(self):
        return dict(case_id='a', trial=0, turn=0, filler=False, success=True, requested_max_tokens=3,
                    ttft=1., tpot_s=.1, latency=1.2, tbt_s=[.1, .1], tbt_complete=True, prompt_len=100)

    def test_slo_matched_requests_and_missing_coverage(self):
        base = self.row()
        load = dict(base, ttft=4.)
        report = s.slo_report([load], {'turns': [base]})
        self.assertEqual(report['state'], 'fail')
        self.assertEqual(report['checks']['ttft']['p50']['value'], 4.)
        self.assertEqual(s.slo_report([load], {'turns': []})['state'], 'unavailable')

    def test_equal_request_tbt_weighting(self):
        # A 1000-token request and a one-gap request have equal total mass.
        values = [(1., .001)] * 1000 + [(10., 1.)]
        self.assertEqual(s.weighted_percentile(values, 90), 10.)

    def test_failed_partial_tokens_excluded_from_compute(self):
        row = dict(start_time=0., end_time=2., success=True, status='success', filler=False,
                   prompt_len=100, output_len=3, retokenized_len=3, ttft=1., latency=2., itl=[.5, .5],
                   tbt_s=[.5, .5], tbt_complete=True, server_metrics=dict(prefill_compute_tokens=10, decode_compute_tokens=2),
                   usage={'prompt_tokens_details': {'cached_tokens': 90}})
        failed = dict(row, success=False, status='timeout')
        report = s.summary([row, failed], 0., 4.)
        self.assertEqual(report['actual']['all_throughput'], 3.)
        self.assertEqual(report['sglang_logical']['total_input'], 100)
        self.assertEqual(report['output_lens'], [3, 0])
        self.assertEqual(report['failed'], 1)


if __name__ == '__main__':
    unittest.main()
