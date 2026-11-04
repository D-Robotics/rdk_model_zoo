# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Target isolation and the independently rebuilt plate input contract."""
import unittest
import numpy as np
from samples.vision.lprnet.runtime.python.cli import BindingError, list_available_assets, resolve_selection
from samples.vision.lprnet.model.prepare_input import prepare_bgr
from samples.vision.lprnet.runtime.python.lprnet import bind_model

class SConversionTests(unittest.TestCase):
    def test_target_isolation_and_exact_external_path(self):
        for target in ('s100', 's100p', 's600'):
            row, = list_available_assets(target)
            self.assertTrue(row.filename.startswith(target + '/'))
            self.assertIn('/rdk_model_zoo_web/rdk_' + target + '/', row.url)
            self.assertEqual(len(row.sha256), 64)
            selected = resolve_selection(target, asset_id=row.reference, model_path='/tmp/model.hbm')
            self.assertEqual(selected.asset, row)
            with self.assertRaises(BindingError):
                resolve_selection(target, model_path='/tmp/model.hbm')
            with self.assertRaises(BindingError):
                resolve_selection('x5', asset_id=row.reference)
            other = 's600' if target != 's600' else 's100'
            with self.assertRaises(BindingError):
                resolve_selection(other, asset_id=row.reference)

    def test_bgr_order_normalization_and_packed_dtype(self):
        image = np.zeros((24, 94, 3), np.uint8)
        image[:] = [0, 127, 255]
        value = prepare_bgr(image)
        self.assertEqual(value.shape, (1, 3, 24, 94))
        self.assertEqual(value.dtype, np.float32)
        self.assertTrue(value.flags.c_contiguous)
        np.testing.assert_array_equal(value[0, :, 0, 0], np.array([-127.5, -.5, 127.5], np.float32) / 128)
        with self.assertRaises(ValueError):
            prepare_bgr(image.astype(np.float32))

    def test_collected_s_abi_binds_complete_logits(self):
        selection = resolve_selection('s100')
        meta = dict(model_names=['lprnet'], model_name='lprnet', input_names=['images'],
                    input_shapes={'images':(1,3,24,94)}, input_dtypes={'images':'float32'},
                    output_names=['logits'], output_shapes={'logits':(1,68,18,1)},
                    output_dtypes={'logits':'float32'})
        binding = bind_model(selection, meta)
        self.assertEqual(binding.output_name, 'logits')
        self.assertEqual(binding.output_shape, (1,68,18,1))

if __name__ == '__main__':
    unittest.main()
