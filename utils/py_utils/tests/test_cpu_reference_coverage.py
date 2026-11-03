"""A truncated CPU reference must never become an empty passing gate."""
from pathlib import Path
import hashlib
import json
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / 'tools/validate_onnx_cpu_reference.py'


class CPUReferenceCoverageTest(unittest.TestCase):
    def run_check(self, case_count=4, status='frozen', output_count=1):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def identity(path):
                return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'size_bytes': path.stat().st_size}

            graph = root / 'model.onnx'
            graph.write_bytes(b'fixture graph')
            manifest = root / 'manifest.json'
            manifest.write_text('{}')
            real = []
            for index in range(3):
                path = root / f'{index}.npy'
                np.save(path, np.zeros((1, 2), np.float32))
                real.append({'input': identity(path)})
            case = {'onnx': identity(graph), 'calibration_manifest': identity(manifest),
                    'real_inputs': real, 'toolchain_image_id': 'fixture'}
            source = root / 'case.json'
            source.write_text(json.dumps(case))
            reference = root / 'reference'
            reference.mkdir()
            records = []
            for index, name in enumerate(['0.npy', '1.npy', '2.npy', 'random-seed0'][:case_count]):
                path = reference / f'{index}.npz'
                np.savez(path, output=np.ones((1, 2), np.float32))
                records.append({'case': name, 'outputs': identity(path)})
            (reference / 'reference.json').write_text(json.dumps({
                'status': status, 'onnx': case['onnx'], 'real_inputs': real, 'cases': records,
                'onnxruntime': 'fixture-reference', 'provider': ['CPUExecutionProvider']}))
            calls = []

            class Session:
                def get_inputs(self):
                    return [types.SimpleNamespace(name='input', shape=[1, 2], type='tensor(float)')]

                def get_outputs(self):
                    return [types.SimpleNamespace(name='output', shape=[1, 2], type='tensor(float)')]

                def get_providers(self):
                    return ['CPUExecutionProvider']

                def run(self, names, feed):
                    calls.append(feed)
                    return [np.ones((1, 2), np.float32)] * output_count

            ort = types.SimpleNamespace(__version__='fixture-sdk', SessionOptions=types.SimpleNamespace,
                                        InferenceSession=lambda *a, **kw: Session())
            output = root / 'result.json'
            with patch.dict(sys.modules, {'onnxruntime': ort}), patch.object(sys, 'argv', [
                str(SCRIPT), '--case', str(source), '--reference', str(reference), '--output', str(output)]):
                try:
                    runpy.run_path(str(SCRIPT), run_name='__main__')
                except SystemExit as error:
                    if error.code != 0:
                        raise
            return calls, json.loads(output.read_text())

    def test_empty_and_truncated_references_are_rejected(self):
        for count in (0, 1, 3):
            with self.subTest(count=count), self.assertRaises(AssertionError):
                self.run_check(case_count=count)

    def test_unfrozen_reference_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.run_check(status='failed')

    def test_missing_and_extra_forward_outputs_are_rejected(self):
        for count in (0, 2):
            with self.subTest(count=count), self.assertRaises(AssertionError):
                self.run_check(output_count=count)

    def test_complete_reference_compares_every_case(self):
        calls, report = self.run_check()
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(report['cases']), 4)
        self.assertTrue(all(len(case['outputs']) == 1 for case in report['cases']))


if __name__ == '__main__':
    unittest.main()
