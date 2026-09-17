import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_serving_matrix as runner


class MatrixTests(unittest.TestCase):
    def test_matrix(self):
        self.assertEqual(runner.MATRIX, [(1, 3), (2, 6), (4, 12), (8, 24), (16, 48), (32, 96)])

    def test_gpu_and_port_isolation(self):
        runner.validate_resources([('a', '0,1', 31080), ('b', '2,3', 31081)])
        for groups in [[('a', '0,1', 1), ('b', '1,2', 2)], [('a', '0,1', 1), ('b', '2,3', 1)],
                       [('a', '0,4', 1)], [('a', '0,0', 1)]]:
            with self.assertRaises(ValueError):
                runner.validate_resources(groups)

    def test_resume_requires_identity_and_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / 'result.json'
            result.write_text(json.dumps({'valid': True}))
            status = root / 'status.json'
            identity = {'head': 'abc', 'mode': 'drop'}
            status.write_text(json.dumps(dict(state='completed', identity=identity,
                                              result=str(result), sha256=runner.bench.file_hash(result))))
            self.assertEqual(runner.reusable(status, identity), result)
            with self.assertRaises(ValueError):
                runner.reusable(status, dict(identity, head='def'))
            result.write_text(json.dumps({'valid': False}))
            with self.assertRaises(ValueError):
                runner.reusable(status, identity)

    def test_single_experiment_uses_only_selected_gpu_pair(self):
        args = SimpleNamespace(phase='experiment', mode='drop-aware', concurrency=32,
                               rounds=3, drop_aware_gpus='0,1', ordinary_gpus='0,1', port=31080)
        self.assertEqual(runner.selected_resources(args), [('drop-aware', '0,1', 31080)])
        args.mode = 'ordinary'
        self.assertEqual(runner.selected_resources(args), [('ordinary', '0,1', 31080)])
        args.phase = 'matrix'
        with self.assertRaises(ValueError):
            runner.selected_resources(args)

    def test_experiment_rejects_unsupported_capacity_and_task_counts(self):
        for c, rounds in [(0, 3), (33, 3), (32, 0), (32, 6)]:
            args = SimpleNamespace(phase='experiment', mode='drop-aware', concurrency=c,
                                   rounds=rounds, drop_aware_gpus='0,1', port=31080)
            with self.assertRaises(ValueError):
                runner.selected_resources(args)


if __name__ == '__main__':
    unittest.main()
