# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline numeric regression checks for the two SAM conversion surfaces.

The source scripts are loaded directly and run with a temporary ``sys.argv``.
The unified copies are called through their ``main(argv)`` entry points.  No
model runtime is involved; the two embedding dump checks inject a recording
fake ONNX Runtime session.
"""

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def _load_module(path: Path, name: str, *, ort_module=None):
    old_ort = sys.modules.get("onnxruntime")
    if ort_module is not None:
        sys.modules["onnxruntime"] = ort_module
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise AssertionError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if old_ort is None:
            sys.modules.pop("onnxruntime", None)
        else:
            sys.modules["onnxruntime"] = old_ort


def _run_legacy(path: Path, argv: list[str], *, ort_module=None):
    module = _load_module(path, f"sam_legacy_{path.stem}_{abs(hash(str(path)))}", ort_module=ort_module)
    old_argv = sys.argv
    try:
        sys.argv = [str(path), *argv]
        with contextlib.chdir(path.parent):
            return module.main()
    finally:
        sys.argv = old_argv


def _run_unified(path: Path, argv: list[str], *, ort_module=None):
    module = _load_module(path, f"sam_unified_{path.stem}_{abs(hash(str(path)))}")
    old_argv = sys.argv
    old_ort = sys.modules.get("onnxruntime")
    try:
        if ort_module is not None:
            sys.modules["onnxruntime"] = ort_module
        sys.argv = [str(path), *argv]
        with contextlib.chdir(path.parent):
            return module.main(argv)
    finally:
        sys.argv = old_argv
        if old_ort is None:
            sys.modules.pop("onnxruntime", None)
        else:
            sys.modules["onnxruntime"] = old_ort


def _write_image(directory: Path) -> Path:
    image = np.array(
        [
            [[0, 10, 20], [30, 40, 50], [60, 70, 80], [90, 100, 110], [120, 130, 140]],
            [[5, 15, 25], [35, 45, 55], [65, 75, 85], [95, 105, 115], [125, 135, 145]],
            [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120], [130, 140, 150]],
        ],
        dtype=np.uint8,
    )
    path = directory / "fixture.png"
    assert cv2.imwrite(str(path), image)
    return path


def _payloads(root: Path, suffix: str, shape: tuple[int, ...], directory_fragment: str | None = None) -> dict[str, np.ndarray]:
    paths = sorted(
        path for path in root.rglob(f"*{suffix}")
        if directory_fragment is None or directory_fragment in path.parent.name
    )
    if not paths:
        raise AssertionError(f"No {suffix} output under {root}")
    result = {}
    for path in paths:
        if path.name in result:
            raise AssertionError(f"Duplicate output basename {path.name}")
        if suffix == ".npy":
            value = np.load(path, allow_pickle=False)
        else:
            value = np.fromfile(path, dtype=np.float32).reshape(shape)
        result[path.name] = value
    return result


def _assert_payloads_equal(test: unittest.TestCase, left, right):
    test.assertEqual(set(left), set(right))
    for name in sorted(left):
        test.assertEqual(left[name].dtype, right[name].dtype, name)
        test.assertEqual(left[name].shape, right[name].shape, name)
        test.assertTrue(np.array_equal(left[name], right[name]), name)


class _RecordingORT(types.ModuleType):
    def __init__(self, embedding):
        super().__init__("onnxruntime")
        self.embedding = embedding
        self.calls = []

        class Session:
            def __init__(session_self, path):
                session_self.path = path

            def run(session_self, output_names, feeds):
                self.calls.append((session_self.path, tuple(output_names), feeds))
                return [self.embedding]

        self.InferenceSession = Session


class SAMConversionRegressionTests(unittest.TestCase):
    def test_all_four_calibration_groups_match_source_values(self):
        image_specs = (
            ("efficient_sam", "x5", "prepare_calibration.py", "prepare_calibration.py", ".rgbchw"),
            ("efficient_sam", "s100", "prepare_calibration.py", "prepare_calibration.py", ".npy"),
            ("mobile_sam", "x5", "prepare_calibration.py", "prepare_calibration.py", ".rgbchw"),
            ("mobile_sam", "s100", "prepare_calibration.py", "prepare_calibration.py", ".npy"),
        )
        for sample, target, source_name, unified_name, suffix in image_specs:
            with self.subTest(sample=sample, target=target):
                with tempfile.TemporaryDirectory(prefix="sam-calibration-") as temp:
                    root = Path(temp)
                    image_dir = root / "images"
                    image_dir.mkdir()
                    image = _write_image(image_dir)
                    source_out = root / "source"
                    unified_out = root / "unified"
                    source_path = ROOT / "platforms" / ("x5" if target == "x5" else "s") / "samples" / "vision" / sample / "conversion" / "scripts" / source_name
                    unified_path = ROOT / "samples" / "vision" / sample / "conversion" / "scripts" / unified_name
                    source_args = ["--src", str(image.parent), "--out", str(source_out), "--num", "20", "--size", "8"]
                    unified_args = ["--target", target, "--src", str(image.parent), "--out", str(unified_out), "--num", "20", "--size", "8"]
                    _run_legacy(source_path, source_args)
                    _run_unified(unified_path, unified_args)
                    shape = (1, 3, 8, 8)
                    _assert_payloads_equal(self, _payloads(source_out, suffix, shape), _payloads(unified_out, suffix, shape))

    def test_all_four_decoder_calibration_groups_match_source_values(self):
        specs = (
            ("efficient_sam", "x5", "prepare_efficient_decoder_calibration.py", "prepare_efficient_decoder_calibration.py", (".bin",), ()),
            ("efficient_sam", "s100", "prepare_efficient_decoder_calibration.py", "prepare_efficient_decoder_calibration.py", (".npy",), ()),
            ("mobile_sam", "x5", "prepare_decoder_calibration.py", "prepare_decoder_calibration.py", (".bin",), (".bin",)),
            ("mobile_sam", "s100", "prepare_decoder_calibration.py", "prepare_decoder_calibration.py", (".npy",), (".npy",)),
        )
        embedding = np.linspace(-2.0, 3.0, 1 * 256 * 32 * 32, dtype=np.float32).reshape(1, 256, 32, 32)
        for sample, target, source_name, unified_name, embedding_suffixes, box_suffixes in specs:
            with self.subTest(sample=sample, target=target):
                with tempfile.TemporaryDirectory(prefix="sam-decoder-calibration-") as temp:
                    root = Path(temp)
                    embedding_path = root / "embedding.bin"
                    embedding.tofile(embedding_path)
                    source_out = root / "source"
                    unified_out = root / "unified"
                    source_path = ROOT / "platforms" / ("x5" if target == "x5" else "s") / "samples" / "vision" / sample / "conversion" / "scripts" / source_name
                    unified_path = ROOT / "samples" / "vision" / sample / "conversion" / "scripts" / unified_name
                    source_args = ["--embedding", str(embedding_path), "--out", str(source_out), "--num", "4"]
                    unified_args = ["--target", target, "--embedding", str(embedding_path), "--out", str(unified_out), "--num", "4"]
                    _run_legacy(source_path, source_args)
                    _run_unified(unified_path, unified_args)
                    for suffix in embedding_suffixes:
                        _assert_payloads_equal(self, _payloads(source_out, suffix, embedding.shape, "embedding"), _payloads(unified_out, suffix, embedding.shape, "embedding"))
                    for suffix in box_suffixes:
                        _assert_payloads_equal(self, _payloads(source_out, suffix, (1, 4), "box"), _payloads(unified_out, suffix, (1, 4), "box"))

    def test_s_dump_embedding_matches_source_and_checks_preprocess_and_raw_save(self):
        embedding = np.linspace(-1.0, 1.0, 1 * 256 * 32 * 32, dtype=np.float32).reshape(1, 256, 32, 32)
        for sample, input_name, normalise in (
            ("efficient_sam", "batched_images", False),
            ("mobile_sam", "normalized_images", True),
        ):
            with self.subTest(sample=sample):
                with tempfile.TemporaryDirectory(prefix="sam-dump-") as temp:
                    root = Path(temp)
                    image_dir = root / "images"
                    image_dir.mkdir()
                    image = _write_image(image_dir)
                    source_out = root / "source.bin"
                    unified_out = root / "unified.bin"
                    onnx_path = root / "encoder.onnx"
                    onnx_path.write_bytes(b"fixture")
                    source_path = ROOT / "platforms" / "s" / "samples" / "vision" / sample / "conversion" / "scripts" / "dump_encoder_embedding.py"
                    unified_path = ROOT / "samples" / "vision" / sample / "conversion" / "scripts" / "dump_encoder_embedding.py"
                    source_ort = _RecordingORT(embedding)
                    unified_ort = _RecordingORT(embedding)
                    args = ["--onnx", str(onnx_path), "--image", str(image), "--output", str(source_out), "--size", "8"]
                    _run_legacy(source_path, args, ort_module=source_ort)
                    args[-3] = str(unified_out)
                    _run_unified(unified_path, args, ort_module=unified_ort)
                    source_value = np.fromfile(source_out, dtype=np.float32).reshape(embedding.shape)
                    unified_value = np.fromfile(unified_out, dtype=np.float32).reshape(embedding.shape)
                    self.assertEqual(source_value.dtype, np.dtype(np.float32))
                    self.assertEqual(unified_value.dtype, np.dtype(np.float32))
                    self.assertTrue(np.array_equal(source_value, embedding))
                    self.assertTrue(np.array_equal(unified_value, embedding))
                    self.assertEqual(source_ort.calls[0][0], str(onnx_path))
                    self.assertEqual(unified_ort.calls[0][0], str(onnx_path))
                    self.assertEqual(source_ort.calls[0][1], ("image_embeddings",))
                    self.assertEqual(unified_ort.calls[0][1], ("image_embeddings",))
                    source_tensor = source_ort.calls[0][2][input_name]
                    unified_tensor = unified_ort.calls[0][2][input_name]
                    self.assertEqual(source_tensor.dtype, np.dtype(np.float32))
                    self.assertEqual(source_tensor.shape, (1, 3, 8, 8))
                    self.assertTrue(np.array_equal(source_tensor, unified_tensor))
                    rgb = cv2.cvtColor(cv2.imread(str(image), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (8, 8), interpolation=cv2.INTER_LINEAR)
                    chw = rgb.transpose(2, 0, 1).astype(np.float32)
                    expected = ((chw - np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)) / np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1))[None] if normalise else chw[None] / 255.0
                    self.assertTrue(np.array_equal(source_tensor, expected))


if __name__ == "__main__":
    unittest.main()
