"""Check catalog-independent model APIs and published tensor behavior."""
from pathlib import Path
import importlib
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
SAMPLES = {'convnext': 'ConvNeXtClassifier', 'edgenext': 'EdgeNeXtClassifier', 'efficientformer': 'EfficientFormerClassifier', 'efficientformerv2': 'EfficientFormerV2Classifier', 'efficientnet': 'EfficientNetClassifier', 'efficientvit': 'EfficientViTClassifier', 'fasternet': 'FasterNetClassifier', 'fastvit': 'FastViTClassifier', 'googlenet': 'GoogLeNetClassifier', 'hgnetv2': 'HGNetV2Classifier', 'mobilenetv1': 'MobileNetV1Classifier', 'mobilenetv2': 'MobileNetV2Classifier', 'mobilenetv3': 'MobileNetV3Classifier', 'mobilenetv4': 'MobileNetV4Classifier', 'mobileone': 'MobileOneClassifier', 'repghost': 'RepGhostClassifier', 'repvgg': 'RepVGGClassifier', 'repvit': 'RepViTClassifier', 'resnext': 'ResNeXtClassifier', 'vargconvnet': 'VargConvNetClassifier', 'vit': 'ViTClassifier'}

class LocalModelInterfaceTests(unittest.TestCase):

    def test_custom_model_predict_is_catalog_independent_for_both_input_protocols(self):
        for sample, name in SAMPLES.items():
            for target in ('x5', 's100'):
                with self.subTest(sample=sample, target=target), tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / ('model.bin' if target == 'x5' else 'model.hbm')
                    path.touch()
                    seen = []

                    class SDK:
                        model_names = ['classifier']
                        input_names = {'classifier': ['data'] if target == 'x5' else ['y', 'uv']}
                        input_shapes = {'classifier': {'data': (1, 3, 32, 48)} if target == 'x5' else {'y': (1, 32, 48, 1), 'uv': (1, 16, 24, 2)}}
                        input_dtypes = {'classifier': {n: 'uint8' for n in input_names['classifier']}}
                        output_names = {'classifier': ['scores']}
                        output_shapes = {'classifier': {'scores': (1, 3)}}
                        output_dtypes = {'classifier': {'scores': 'float32'}}

                        def run(self, payload):
                            seen.append(payload)
                            return {'classifier': {'scores': np.array([[0.25, 3.0, -2.0]], dtype=np.float32)}}
                    model_class = getattr(importlib.import_module(f'samples.vision.{sample}.runtime.python.classify'), name)
                    with patch('utils.py_utils.platforms.require_execution_target'), patch('utils.py_utils.runtime._default_runtime_factory', return_value=lambda _: SDK()), patch('utils.py_utils.cls_binding.read_manifest_asset_records', side_effect=AssertionError('catalog access')):
                        model = model_class(path, target=target, input_size=(32, 48), class_count=3, top_k=2)
                        result = model.predict(np.full((25, 40, 3), 127, dtype=np.uint8))
                    self.assertEqual(result.class_ids.tolist(), [1, 0])
                    self.assertEqual(len(seen), 1)
                    shapes = sorted((a.shape for a in seen[0]['classifier'].values()))
                    self.assertEqual(shapes, [(2304,)] if target == 'x5' else [(1, 16, 24, 2), (1, 32, 48, 1)])

    def test_published_variants_preserve_tensor_bytes_and_topk(self):
        """Compare path-based models with the existing contract-based pipeline."""
        from utils.py_utils.classification import ClassificationTask
        from utils.py_utils.model_runner import RuntimeModelRunner

        image = (np.arange(37 * 61 * 3) % 256).astype(np.uint8).reshape(37, 61, 3)
        for sample, name in SAMPLES.items():
            cli = importlib.import_module(f"samples.vision.{sample}.runtime.python.cli")
            model_class = getattr(importlib.import_module(
                f"samples.vision.{sample}.runtime.python.classify"), name)
            for record in cli.list_available_assets():
                with self.subTest(sample=sample, asset=record.asset_id):
                    selection = cli.resolve_selection(record.target, asset_id=record.asset_id)
                    contract = selection.contract
                    height, width = contract.input_height, contract.input_width
                    physical_shapes = ({"data": (1, 3, height, width)}
                        if record.target == "x5" else {
                            "y": (1, height, width, 1), "uv": (1, height // 2, width // 2, 2)})
                    raw = np.linspace(-3, 4, contract.class_count, dtype=np.float32)[None, :]

                    class SDK:
                        model_names = ["classifier"]
                        input_names = {"classifier": list(physical_shapes)}
                        input_shapes = {"classifier": physical_shapes}
                        input_dtypes = {"classifier": {n: "uint8" for n in physical_shapes}}
                        output_names = {"classifier": ["scores"]}
                        output_shapes = {"classifier": {"scores": raw.shape}}
                        output_dtypes = {"classifier": {"scores": "float32"}}

                        def run(self, tensors):
                            return {"classifier": {"scores": raw.copy()}}

                    runner = RuntimeModelRunner(selection, table=cli.BINDING_TABLE, runtime=SDK())
                    old = ClassificationTask(runner, runner.load(), top_k=5)
                    direct = RuntimeModelRunner.from_file(
                        selection.model_path, target=selection.target,
                        input_size=(height, width), class_count=contract.class_count,
                        resize_type=contract.resize_type,
                        resize_interpolation=contract.resize_interpolation,
                        score_policy=contract.output_score_policy,
                        output_transform=contract.output_transform, runtime=SDK())
                    model = model_class(selection.model_path, target=selection.target, runner=direct)
                    old_input = old.pre_process(image)
                    new_input = model.preprocess(image)
                    self.assertEqual(set(old_input.tensors), set(new_input.tensors))
                    for tensor_name in old_input.tensors:
                        np.testing.assert_array_equal(old_input.tensors[tensor_name],
                                                      new_input.tensors[tensor_name])
                    before, after = old.predict(image), model.predict(image)
                    np.testing.assert_array_equal(before.class_ids, after.class_ids)
                    np.testing.assert_array_equal(before.scores, after.scores)
                    self.assertEqual(before.labels, after.labels)

    def test_model_free_commands_do_not_import_numpy_opencv_or_sdk(self):
        """CLI help and model discovery work without execution dependencies."""
        import subprocess
        import sys
        root = Path(__file__).resolve().parents[3]
        guard = """import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'cv2', 'hbm_runtime'):
            raise AssertionError('Unexpected execution import: ' + fullname)
sys.meta_path.insert(0, Guard())
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
"""
        for sample in SAMPLES:
            main = root / f"samples/vision/{sample}/runtime/python/main.py"
            for args in (("--help",), ("--list-models",), ("--dry-run",)):
                with self.subTest(sample=sample, args=args), tempfile.TemporaryDirectory() as temp:
                    run = subprocess.run([sys.executable, "-c", guard, str(main), *args],
                                         cwd=temp, capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, run.stderr)

if __name__ == '__main__':
    unittest.main()
