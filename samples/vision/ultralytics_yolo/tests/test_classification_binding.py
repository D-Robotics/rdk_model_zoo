"""Classification uses validated raw transport, source math and owned results."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from scipy.special import softmax
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    ClassificationContract,
    ModelSelection,
    RuntimeMetadata,
    bind_model,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import ModelRunner
from samples.vision.ultralytics_yolo.runtime.python.yolo_cls import (
    YoloCls,
    YoloClsConfig,
)


def fixture(target="s100", shape=(1, 1000), dtype=np.float32):
    value = np.linspace(-10, 10, 1000, dtype=dtype).reshape(shape)
    ins = (
        {"image": (1, 3, 224, 224)}
        if target == "x5"
        else {"y": (1, 224, 224, 1), "uv": (1, 112, 112, 2)}
    )
    meta = RuntimeMetadata(
        "m",
        tuple(ins),
        ins,
        ("opaque",),
        {"opaque": shape},
        {k: np.uint8 for k in ins},
        {"opaque": dtype},
    )
    selection = ModelSelection(
        "stub", target=target, task="classify", contract=ClassificationContract()
    )
    binding = bind_model(selection, meta)
    sdk = SimpleNamespace(run=Mock(return_value={"m": {"opaque": value}}))
    runner = ModelRunner(sdk, binding, meta)
    return YoloCls(YoloClsConfig("stub"), runner=runner), value, sdk, meta, selection


class ClassificationBinding(unittest.TestCase):
    def test_all_targets_raw_identity_and_source_math(self):
        for target in ("x5", "s100", "s100p", "s600"):
            model, values, sdk, _, _ = fixture(target)
            inputs = model.pre_process(np.zeros((73, 151, 3), np.uint8))
            self.assertEqual(
                sum(v.size for v in inputs["m"].values()), 224 * 224 * 3 // 2
            )
            raw = model.forward(inputs)
            self.assertIs(raw["logits"], values)
            sdk.run.assert_called_once_with(inputs)
            result = model.post_process(raw)
            probs = softmax(values.reshape(-1))
            ids = np.argsort(probs)[::-1][:5]
            self.assertEqual(result, [(int(i), float(probs[i])) for i in ids])
            values.fill(0)
            self.assertEqual(result[0][0], 999)
            self.assertGreater(result[0][1], result[-1][1])

    def test_predict_and_stages_agree_and_limits_are_explicit(self):
        model, _, sdk, _, _ = fixture()
        img = np.zeros((51, 29, 3), np.uint8)
        self.assertEqual(
            model.predict(img),
            model.post_process(model.forward(model.pre_process(img))),
        )
        self.assertEqual(len(model.predict(img, topk=1005)), 1000)
        for bad in (0, -1, True, 1.5, "5"):
            with self.subTest(topk=bad), self.assertRaises((TypeError, ValueError)):
                model.post_process(model.forward({}), topk=bad)
        for bad in (
            np.zeros((2, 2), np.uint8),
            np.zeros((2, 2, 3)),
            np.zeros((0, 2, 3), np.uint8),
        ):
            with self.assertRaises(ValueError):
                model.pre_process(bad)
        with self.assertRaises(ValueError):
            model.pre_process(img, "RGB")

    def test_metadata_rejects_ambiguous_or_nonfloating_outputs(self):
        _, _, _, meta, selection = fixture()
        cases = [
            replace(meta, output_names=("opaque", "extra")),
            replace(meta, output_shapes={"opaque": (2, 500)}),
            replace(meta, output_shapes={"opaque": (1000, 1)}),
            replace(meta, output_shapes={"opaque": (1, 999)}),
            replace(meta, output_shapes={}),
            replace(meta, output_dtypes={}),
            replace(meta, output_dtypes={"opaque": np.int32}),
            replace(
                meta,
                output_quantization={"opaque": {"quant_type": "SCALE", "scale": 1}},
            ),
        ]
        for bad in cases:
            with self.subTest(metadata=bad), self.assertRaises(ValueError):
                bind_model(selection, bad)
        for shape in ((1000,), (1, 1000), (1, 1, 1, 1000), (1, 1000, 1, 1)):
            model, _, _, _, _ = fixture(shape=shape)
            self.assertEqual(model.predict(np.zeros((12, 14, 3), np.uint8))[0][0], 999)

    def test_buffer_contract_and_foreign_binding(self):
        model, values, sdk, _, _ = fixture()
        other, _, _, _, _ = fixture()
        with self.assertRaises(ValueError):
            other.post_process(model.forward({}))
        values[0, 0] = np.nan
        with self.assertRaises(RuntimeError):
            model.forward({})
        sdk.run.return_value = {"m": {"opaque": np.zeros((1, 999), np.float32)}}
        with self.assertRaises(RuntimeError):
            model.forward({})


if __name__ == "__main__":
    unittest.main()
