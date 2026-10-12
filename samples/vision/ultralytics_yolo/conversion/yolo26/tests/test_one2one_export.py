"""Real-weight regression: YOLO26_TEST_WEIGHTS points to official nano .pt files.

Run with Python unittest discovery in an environment with ultralytics, onnx,
onnxruntime and torch installed (validated with ultralytics==8.4.153).
Exports must match the original one-to-one heads.

Example, from the conversion/yolo26 directory:
    YOLO26_TEST_WEIGHTS=/path/to/weights python -m unittest discover -s tests -v
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

# Optional-scope declaration: module-level framework imports let the host
# gate record absent torch/ultralytics as explicit missing-module scope.
import torch  # noqa: F401
from ultralytics import YOLO  # noqa: F401


@unittest.skipUnless(os.environ.get('YOLO26_TEST_WEIGHTS'), 'Set YOLO26_TEST_WEIGHTS to the weights directory')
class OneToOneExportTest(unittest.TestCase):
    def test_real_weight_exports(self):
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(4)
        scripts = Path(__file__).resolve().parents[1]
        weights = Path(os.environ['YOLO26_TEST_WEIGHTS']).resolve()
        image = np.random.default_rng(176).random((1, 3, 640, 640), dtype=np.float32)
        for task, suffix in [('detect', ''), ('seg', '-seg'), ('pose', '-pose'), ('obb', '-obb')]:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                source = weights / f'yolo26n{suffix}.pt'
                self.assertTrue(source.is_file(), str(source))
                checkpoint = Path(tmp) / source.name
                shutil.copyfile(source, checkpoint)
                output = Path(tmp) / 'export.onnx'
                result = subprocess.run(
                    [sys.executable, str(scripts / f'export_yolo26_{task}_bpu.py'),
                     '--weights', str(checkpoint), '--output', str(output), '--imgsz', '640', '--platform', 's600'],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(output.is_file(), result.stdout + result.stderr)
                onnx.checker.check_model(onnx.load(str(output)))
                session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
                actual = session.run(None, {session.get_inputs()[0].name: image})

                # Independent oracle: native model, explicit one-to-one weights,
                # no imports of the patched exporter or its forward function.
                reference = YOLO(str(source)).model.float().eval()
                reference.end2end = True
                reference.fuse()
                head = reference.model[-1]
                features = []
                hook = head.register_forward_pre_hook(
                    lambda module, args: features.extend(t.clone() for t in args[0])
                )
                with torch.no_grad():
                    reference(torch.from_numpy(image))
                    hook.remove()
                    expected = []
                    for i, feature in enumerate(features):
                        expected.extend([head.one2one_cv3[i](feature), head.one2one_cv2[i](feature)])
                        if task in ('seg', 'obb'):
                            expected.append(head.one2one_cv4[i](feature))
                        elif task == 'pose':
                            expected.append(head.one2one_cv4_kpts[i](head.one2one_cv4[i](feature)))
                    if task == 'seg':
                        proto = head.proto(features)
                        expected.append(proto[0] if isinstance(proto, (tuple, list)) else proto)
                    expected = [t.permute(0, 2, 3, 1).numpy() for t in expected]
                self.assertEqual(len(actual), len(expected))
                errors = []
                for value, target in zip(actual, expected):
                    self.assertEqual(value.shape, target.shape)
                    self.assertTrue(np.isfinite(value).all())
                    np.testing.assert_allclose(value, target, rtol=1e-3, atol=2e-4)
                    errors.append(float(np.max(np.abs(value - target))))
                print(f'{task}: one-to-one match; shapes={[v.shape for v in actual]}; max_abs={max(errors):.8g}', flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
