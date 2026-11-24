"""Regression checks for the independent upstream base224 input contract."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from PIL import Image
from samples.vision.siglip.runtime.python.public_preprocess import prepare_image
from utils.tools.hf_transformer_conversion import FACTS, compare, export


class PublicSourceTests(unittest.TestCase):
    def test_direct_resize_does_not_add_gray_letterbox(self):
        value = prepare_image(Image.new('RGB', (31, 7), 'white'))
        self.assertEqual(value.shape, (1, 3, 224, 224))
        self.assertEqual(value.dtype, np.float32)
        np.testing.assert_array_equal(value, np.ones_like(value))

    def test_black_image_uses_upstream_normalized_range(self):
        value = prepare_image(Image.new('RGB', (7, 31), 'black'))
        np.testing.assert_array_equal(value, -np.ones_like(value))

    def test_full_output_gate_rejects_a_missing_patch_head(self):
        facts = FACTS['siglip']
        reference = {k: np.zeros(shape, np.float32) for k, shape in facts['outputs'].items()}
        with self.assertRaises(ValueError):
            compare({'pooler_output': reference['pooler_output']}, reference, facts)

    def test_full_output_gate_rejects_an_implicit_double_output(self):
        facts = FACTS['siglip']
        reference = {k: np.zeros(shape, np.float32) for k, shape in facts['outputs'].items()}
        actual = dict(reference)
        actual['last_hidden_state'] = actual['last_hidden_state'].astype(np.float64)
        with self.assertRaises(ValueError):
            compare(actual, reference, facts)

    def test_runtime_retains_both_heads_before_patch_visualization(self):
        from samples.vision.siglip.runtime.python.public import validate_outputs, patch_projection
        values={'pooler_output':np.ones((1,768),np.float32),
                'last_hidden_state':np.zeros((1,196,768),np.float32)}
        self.assertIs(validate_outputs(values),values)
        rgb,details=patch_projection(values)
        self.assertEqual(rgb.shape,(14,14,3))
        self.assertFalse(details['semantic_labels'])
        with self.assertRaises(ValueError):validate_outputs({'pooler_output':values['pooler_output']})

    def test_export_checks_actual_calibration_and_validation_counts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root / 'calibration.json'
            validation = root / 'validation.json'
            for kind, count in [('asr', 50), ('siglip', 200)]:
                def write_manifest(path, purpose, declared, actual):
                    path.write_text(json.dumps({
                        'sample': kind, 'purpose': purpose, 'count': declared,
                        'official_preprocessing_byte_exact': True,
                        'records': [{}] * actual,
                    }))

                for actual in (1, count + 1):
                    with self.subTest(kind=kind, calibration_count=actual):
                        write_manifest(calibration, 'calibration', count, actual)
                        with self.assertRaisesRegex(ValueError, f'{count} verified {kind} calibration'):
                            export(kind, root, calibration, root / 'export')
                write_manifest(calibration, 'calibration', count, count)
                for actual in (0, 9):
                    with self.subTest(kind=kind, validation_count=actual):
                        write_manifest(validation, 'validation', 8, actual)
                        with self.assertRaisesRegex(ValueError, f'8 verified {kind} validation'):
                            export(kind, root, calibration, root / 'export', validation)
