"""Native S YOLOE heads: physical validation, SCALE arithmetic and DFL math."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np

from utils.py_utils import postprocess as post
from samples.vision.ultralytics_yolo.runtime.python.backend import RuntimeMetadata
from samples.vision.yoloe.runtime.python.cli import resolve_selection
from samples.vision.yoloe.runtime.python.yoloe import runtime_selection
from samples.vision.yoloe.runtime.python.yoloe import bind_yoloe_outputs, YOLOERunner
from samples.vision.yoloe.runtime.python.yoloe import YOLOE


def native_metadata(variant):
    """Use the published S100 names/types with nontrivial per-channel scales."""
    e26 = variant.startswith("26")
    names = (["cls_8", "box_8", "mc_8", "cls_16", "box_16", "mc_16", "cls_32", "box_32", "mc_32", "proto"] if e26 else
             ["output0", "output1", "560", "574", "582", "590", "604", "612", "620", "631"])
    channels = [4585, 4 if e26 else 64, 32] * 3 + [32]
    shapes = {n: (1, 640 // stride, 640 // stride, c)
              for n, stride, c in zip(names, [8]*3 + [16]*3 + [32]*3 + [4], channels)}
    dtypes, quants = {}, {}
    for index, (name, c) in enumerate(zip(names, channels)):
        dtype = np.float32 if not e26 and index in (0, 3, 6) else (np.int8 if e26 else np.int16) if index == 9 else np.int32
        dtypes[name] = dtype
        if dtype != np.float32:
            quants[name] = SimpleNamespace(quant_type="SCALE", scale=np.array([0.125], np.float32) if index == 9 else np.arange(1, c+1, dtype=np.float32)/128,
                                          zero_point=[2], axis=0 if index == 9 else 3)
    return RuntimeMetadata("model", ("images_uv", "images_y"),
                           {"images_y": (1,640,640,1), "images_uv": (1,320,320,2)}, tuple(reversed(names)), shapes,
                           {"images_y": np.uint8, "images_uv": np.uint8}, dtypes, quants)


class QuantizedRuntimeTests(unittest.TestCase):
    def test_mixed_e11_and_per_channel_e26_numeric_dequantization(self):
        for variant in ("11s", "26n"):
            metadata = native_metadata(variant)
            binding = bind_yoloe_outputs(runtime_selection(resolve_selection("s100", variant=variant)), metadata, allow_integer=True)
            values = {n: np.full(sh, 10, metadata.output_dtypes[n]) for n, sh in metadata.output_shapes.items()}
            raw = binding.read_raw_outputs({"model": values})
            self.assertEqual(raw["box_8"].dtype, np.int32)
            logical = binding.read_outputs(raw)
            np.testing.assert_array_equal(logical["box_8"][0,0,0], np.arange(1, logical["box_8"].shape[-1]+1, dtype=np.float32)/16)
            np.testing.assert_array_equal(logical["protos"], np.ones((1,160,160,32), np.float32))
            if variant == "11s":
                np.testing.assert_array_equal(logical["cls_8"], values["output0"])
            else:
                self.assertEqual(logical["cls_8"][0,0,0,7], 0.5)
            wrong = dict(values)
            name = binding.output_roles["box_8"]
            wrong[name] = values[name].astype(np.float32)
            with self.assertRaises(ValueError):
                binding.read_raw_outputs({"model": wrong})

    def test_native_e26_predict_keeps_one_sigmoid(self):
        selection = resolve_selection("s100", variant="26n")
        metadata = native_metadata("26n")
        binding = bind_yoloe_outputs(runtime_selection(selection), metadata, allow_integer=True)
        values = {n: np.full(sh, 18, metadata.output_dtypes[n])
                  for n, sh in metadata.output_shapes.items()}
        for stride in (8, 16, 32):
            values[f"cls_{stride}"].fill(-640)
        # Channel 7 scale is 8/128: (98 - zero_point 2) / 16 == logit 6.
        values["cls_8"][0,30,30,7] = 98
        runtime = SimpleNamespace(run=lambda inputs: {"model": values})
        task = YOLOE(selection, runner=YOLOERunner(runtime, binding, metadata))
        result = task.predict(np.zeros((640,640,3), np.uint8))
        np.testing.assert_array_equal(result.class_ids, [7])
        np.testing.assert_allclose(result.scores, [1/(1+np.exp(-6))], rtol=1e-6)
        np.testing.assert_allclose(result.boxes, [[243,242,247,248]])

    def test_dfl_uses_dequantized_logits(self):
        metadata = native_metadata("11s")
        binding = bind_yoloe_outputs(runtime_selection(resolve_selection("s100", variant="11s")), metadata, allow_integer=True)
        values = {n: np.zeros(sh, metadata.output_dtypes[n]) for n, sh in metadata.output_shapes.items()}
        values["output1"][0,0,0] = np.arange(64, dtype=np.int32) + 2
        logical = binding.read_outputs({"model": values})
        box = post.decode_boxes(logical["box_8"], np.array([0]), 80, 8, np.arange(16, dtype=np.float32)[None,None,:])
        logits = np.arange(64, dtype=np.float64).reshape(4,16) * np.arange(1,65).reshape(4,16)/128
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        distances = (probabilities * np.arange(16)).sum(axis=1)/probabilities.sum(axis=1)
        expected = np.array([[4-distances[0]*8, 4-distances[1]*8, 4+distances[2]*8, 4+distances[3]*8]])
        np.testing.assert_allclose(box, expected, rtol=1e-6, atol=1e-5)

    def test_rejects_wrong_shape_dtype_and_quantization(self):
        metadata = native_metadata("26n")
        selected = runtime_selection(resolve_selection("s100", variant="26n"))
        changes = [replace(metadata, output_shapes={**metadata.output_shapes, "box_8": (1,80,80,5)}),
                   replace(metadata, output_dtypes={**metadata.output_dtypes, "box_8": np.int16})]
        for info in (None, SimpleNamespace(quant_type="NONE", scale=[1], zero_point=[], axis=3),
                     SimpleNamespace(quant_type="SCALE", scale=[0], zero_point=[], axis=3),
                     SimpleNamespace(quant_type="SCALE", scale=[1,2,3,4], zero_point=[], axis=1)):
            changes.append(replace(metadata, output_quantization={**metadata.output_quantization, "box_8": info}))
        for bad in changes:
            with self.assertRaises(ValueError):
                bind_yoloe_outputs(selected, bad, allow_integer=True)
        with self.assertRaises(ValueError):
            bind_yoloe_outputs(selected, metadata, allow_integer=False)
