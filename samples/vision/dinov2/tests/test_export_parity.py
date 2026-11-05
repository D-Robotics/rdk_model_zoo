# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exercise the export acceptance gate without importing the PyTorch runtime."""
import ast
from pathlib import Path
import unittest

import numpy as np


class ExportParityTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'conversion/onnx_export/export_dinov2.py'
        tree = ast.parse(path.read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'check_parity')
        scope = {'np': np}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
        self.check = scope['check_parity']
        self.reference = [np.ones((1, 384), np.float32), np.ones((1, 256, 384), np.float32)]

    def test_rejects_missing_outputs_and_nonfinite_values(self):
        for actual in (self.reference[:1], [self.reference[0], self.reference[1] * np.nan]):
            with self.assertRaises(ValueError):
                self.check(self.reference, actual, stage='fixture', sample='real_image')

    def test_rejects_numerical_mismatch_and_accepts_close_outputs(self):
        with self.assertRaises(AssertionError):
            self.check(self.reference, [x * 1.1 for x in self.reference], stage='fixture', sample='real_image')
        rows = self.check(self.reference, [x + 1e-5 for x in self.reference], stage='fixture', sample='real_image')
        self.assertEqual([r['output'] for r in rows], ['cls_feat', 'patch_feat'])

    def test_rejects_matching_but_incorrect_shapes(self):
        with self.assertRaises(ValueError):
            self.check([np.ones(1), np.ones(1)], [np.ones(1), np.ones(1)], stage='fixture', sample='real_image')


class ConversionRecipeTests(unittest.TestCase):
    def test_historical_default_and_explicit_o1_change_only_requested_fields(self):
        import argparse
        import logging
        import os
        import tempfile
        from utils.py_utils.tests.legacy_platforms import legacy_path

        sample = Path(__file__).resolve().parents[1]
        paths = [legacy_path('s/samples/vision/dinov2/conversion/mapper.py'), sample/'conversion/mapper.py']
        scopes = []
        for path in paths:
            tree = ast.parse(path.read_text())
            quant = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'QUANT_CONFIG' for t in n.targets))
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('generate_yaml', 'output_prefix')]
            scope = {'argparse': argparse, 'os': os, 'logger': logging.getLogger(__name__), 'QUANT_CONFIG': ast.literal_eval(quant.value)}
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)
            scopes.append(scope)
        with tempfile.TemporaryDirectory() as directory:
            old, new = scopes
            for march in ('nash-e', 'nash-m', 'nash-p'):
                options = argparse.Namespace(march=march, jobs=2)
                previous = Path(old['generate_yaml'](options, '/source/model.onnx', '/source/calibration', directory)).read_text()
                current = Path(new['generate_yaml'](options, '/source/model.onnx', '/source/calibration', directory)).read_text()
                self.assertEqual(previous, current)
                prefix = f"dinov2_vits14_224_int16_{march.replace('-', '')}"
                options.optimize_level = 'O1'
                options.output_prefix = prefix + '_20261102_o1'
                o1 = Path(new['generate_yaml'](options, '/source/model.onnx', '/source/calibration', directory)).read_text()
                expected = previous.replace("optimize_level: 'O2'", "optimize_level: 'O1'").replace(f"output_model_file_prefix: '{prefix}'", f"output_model_file_prefix: '{options.output_prefix}'")
                self.assertEqual(o1, expected)
