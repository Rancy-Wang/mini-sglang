import json
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == '__main__':
    unittest.main()
