"""Exercise the X5/S calibration boundary with actual written tensor bytes."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from utils.tools.classification_conversion.prepare_calibration import prepare


class ConversionCalibrationTests(unittest.TestCase):
    def test_runtime_normalization_is_applied_to_s_calibration_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);images=root/'images';images.mkdir()
            Image.new('RGB',(224,224),(255,128,0)).save(images/'constant.png')
            norm={'mean':[.5,.25,.125],'std':[.25,.5,1.]}
            x5=prepare(images,root/'x5',1)
            np.testing.assert_array_equal(np.fromfile(root/'x5/0000.rgb',dtype=np.float32).reshape(3,224,224)[:,0,0], [255,128,0])
            self.assertEqual(x5['tensor_domain'],'raw-rgb')
            expected=np.asarray([2.,(128/255-.25)/.5,-.125])
            for target in ('s100','s100p','s600'):
                manifest=prepare(images,root/target,1,target=target,normalization=norm)
                actual=np.load(root/target/'0000.npy',allow_pickle=False)
                self.assertEqual(actual.shape,(1,3,224,224))
                self.assertEqual(actual.dtype,np.float32)
                np.testing.assert_allclose(actual[0,:,0,0],expected,atol=1e-7)
                self.assertEqual(manifest['normalization'],norm)
                self.assertEqual(manifest['tensor_domain'],'normalized-onnx-input')

    def test_missing_normalization_fails_before_creating_s_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ValueError,'checkpoint mean and std'):
                prepare(root,root/'output',1,target='s100')
            self.assertFalse((root/'output').exists())
