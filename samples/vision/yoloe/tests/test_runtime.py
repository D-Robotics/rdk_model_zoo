"""Host selection/metadata/stage checks; fake SDK execution is not board evidence."""

from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np

from samples.vision.yoloe.runtime.python.model_binding import (
    resolve_selection,
    list_models,
)
from samples.vision.yoloe.runtime.python.model_runner import build_runner
from samples.vision.yoloe.runtime.python.yoloe import YOLOE, Config

ROOT = Path(__file__).resolve().parents[4]


def fake_runtime(selection, integer=False):
    names, shapes, data = [], {}, {}
    for stride in (8, 16, 32):
        g = 640 // stride
        for kind, ch in [
            ("cls", 4585),
            ("box", 4 if selection.variant.startswith("26") else 64),
            ("mces", 32),
        ]:
            key = f"{kind}_{stride}"
            names.append(key)
            shapes[key] = (1, g, g, ch)
            data[key] = np.full(shapes[key], -30 if kind == "cls" else 1, np.float32)
    names.append("protos")
    shapes["protos"] = (1, 160, 160, 32)
    data["protos"] = np.ones(shapes["protos"], np.float32)
    data["cls_8"][0, 30, 30, 7] = 6
    data["cls_8"][0, 30, 31, 9] = 5
    ins = (
        {"input": (1, 3, 640, 640)}
        if selection.target == "x5"
        else {"y": (1, 640, 640, 1), "uv": (1, 320, 320, 2)}
    )

    class Runtime:
        model_names = ["m"]
        input_names = {"m": list(ins)}
        input_shapes = ins
        input_dtypes = {n: np.uint8 for n in ins}
        output_names = {"m": list(reversed(names))}
        output_shapes = shapes
        output_dtypes = {n: np.int32 if integer else np.float32 for n in names}
        calls = 0

        def run(self, tensors):
            self.calls += 1
            self.last_input = tensors
            return {"m": data}

    return Runtime(), data


class RuntimeTests(unittest.TestCase):
    def test_published_matrix_and_defaults(self):
        self.assertEqual(len(list_models()), 14)
        for target, variant in [("x5", "11s"), ("s100", "11s"), ("s100p", "26n")]:
            selection = resolve_selection(target)
            self.assertEqual(selection.variant, variant)
            self.assertEqual(selection.published_float, target == "x5")
        for target, variant in [
            ("s600", "26n"),
            ("s100p", "11s"),
            ("x5", "26n"),
            ("s100", "11m"),
        ]:
            with self.assertRaises(ValueError):
                resolve_selection(target, variant=variant)

    def test_exact_asset_and_custom_float_identity(self):
        s = resolve_selection("x5", variant="11m")
        self.assertEqual(
            resolve_selection("x5", asset_id=s.asset.reference).variant, "11m"
        )
        for kwargs in [
            dict(model_path="other.bin"),
            dict(variant="11s", asset_id=s.asset.reference),
            dict(local_float_sha256="f" * 64),
        ]:
            with self.assertRaises(ValueError):
                resolve_selection("x5", **kwargs)
        custom = resolve_selection(
            "s100", variant="26s", model_path="float.hbm", local_float_sha256="a" * 64
        )
        self.assertTrue(custom.local_float)
        self.assertFalse(custom.published_float)

    def test_published_quantized_route_checks_identity_and_asset_before_sdk(self):
        s = resolve_selection("s100")
        with patch("utils.py_utils.platforms.require_execution_target") as gate:
            with self.assertRaisesRegex(ValueError, "Missing or empty model"):
                build_runner(s)
            gate.assert_called_once_with("s100")

    def test_binding_orders_by_shape_and_rejects_integer(self):
        for target, variant in [("x5", "11s"), ("s100", "26n")]:
            s = resolve_selection(target, variant=variant)
            runtime, data = fake_runtime(s)
            runner = build_runner(
                s,
                runtime_loader=lambda: SimpleNamespace(HB_HBMRuntime=lambda p: runtime),
            )
            self.assertEqual(runner.contract.classes, 4585)
            self.assertEqual(
                runner.binding.output_adapter.role_to_name["cls_8"], "cls_8"
            )
            self.assertEqual(
                runner.contract.nms, "none" if variant.startswith("26") else "classwise"
            )
            runtime.output_dtypes = {n: np.int32 for n in data}
            with self.assertRaises((ValueError, RuntimeError)):
                build_runner(
                    s,
                    runtime_loader=lambda: SimpleNamespace(
                        HB_HBMRuntime=lambda p: runtime
                    ),
                )

    def test_three_stages_match_predict_and_own_results(self):
        for target, variant in [("x5", "11s"), ("s100", "11s"), ("s100p", "26n")]:
            s = resolve_selection(target, variant=variant)
            runtime, data = fake_runtime(s)
            runner = build_runner(
                s,
                runtime_loader=lambda: SimpleNamespace(HB_HBMRuntime=lambda p: runtime),
            )
            task = YOLOE(s, runner=runner)
            image = np.zeros((320, 640, 3), np.uint8)
            prepared = task.pre_process(image)
            raw = task.forward(prepared)
            self.assertEqual(runtime.calls, 1)
            actual = task.post_process(raw, prepared.context)
            result = task.predict(image)
            for name in ("boxes", "scores", "class_ids"):
                np.testing.assert_array_equal(
                    getattr(actual, name), getattr(result, name)
                )
            self.assertEqual(result.mask_layout, "full" if target == "x5" else "roi")
            self.assertEqual(len(result.masks), len(result.boxes))
            self.assertGreater(len(result.boxes), 0)
            snapshots = [mask.copy() for mask in result.masks]
            data["protos"].fill(-1)
            for a, b in zip(result.masks, snapshots):
                np.testing.assert_array_equal(a, b)
            with self.assertRaises(ValueError):
                task.pre_process(image.astype(np.float32))

    def test_protocol_specific_option_validation(self):
        s = resolve_selection("s100", variant="26n")
        runtime, _ = fake_runtime(s)
        runner = build_runner(
            s, runtime_loader=lambda: SimpleNamespace(HB_HBMRuntime=lambda p: runtime)
        )
        for cfg in [
            Config(nms_thres=0.7),
            Config(resize_type=0),
            Config(do_morph=True),
            Config(max_det=True),
            Config(score_thres=1),
        ]:
            with self.assertRaises(ValueError):
                YOLOE(s, cfg, runner=runner)

    def test_cli_host_dry_run_and_matrix(self):
        main = ROOT / "samples/vision/yoloe/runtime/python/main.py"
        for target, variant in [("x5", "11s"), ("s100", "26n"), ("s100p", "26x")]:
            result = subprocess.run(
                [
                    sys.executable,
                    str(main),
                    "--target",
                    target,
                    "--variant",
                    variant,
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            facts = json.loads(result.stdout)
            self.assertEqual(facts["variant"], variant)
            self.assertEqual(facts["published_float"], target == "x5")
        result = subprocess.run(
            [sys.executable, str(main), "--target", "s600", "--dry-run"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)


class BoundaryTests(unittest.TestCase):
    def test_semantic_dictionary_cannot_bypass_tensor_validation(self):
        s = resolve_selection("x5")
        runtime, data = fake_runtime(s)
        runner = build_runner(
            s, runtime_loader=lambda: SimpleNamespace(HB_HBMRuntime=lambda p: runtime)
        )
        task = YOLOE(s, runner=runner)
        context = task.pre_process(np.zeros((320, 640, 3), np.uint8)).context
        for bad in (
            dict(data, unexpected=np.zeros(1)),
            dict(data, protos=data["protos"].astype(np.int32)),
            dict(data, box_8=data["box_8"][:, :, :1]),
        ):
            with self.assertRaises(ValueError):
                task.post_process(bad, context)

    def test_cli_s11_morph_default_preserves_source(self):
        from samples.vision.yoloe.runtime.python.main import main
        from contextlib import redirect_stdout
        import io

        for target, variant, expected in [
            ("s100", "11s", True),
            ("x5", "11s", False),
            ("s100p", "26n", False),
        ]:
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(
                    main(["--target", target, "--variant", variant, "--dry-run"]), 0
                )
            self.assertEqual(json.loads(out.getvalue())["config"]["do_morph"], expected)

    def test_custom_digest_checked_before_loading(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.hbm"
            path.write_bytes(b"not a real model")
            s = resolve_selection("s100", model_path=path, local_float_sha256="0" * 64)
            with patch("utils.py_utils.platforms.require_execution_target"), patch(
                "utils.py_utils.runtime.RuntimeSession.load"
            ) as load:
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    build_runner(s)
                load.assert_not_called()

    def test_vocabulary_and_visualization(self):
        import tempfile
        import cv2
        from samples.vision.yoloe.runtime.python.visualization import (
            load_inputs,
            save_result,
        )
        from samples.vision.yoloe.runtime.python.yoloe import Result

        sample = ROOT / "samples/vision/yoloe"
        image, labels = load_inputs(
            sample / "test_data/office_desk.jpg", sample / "test_data/classes.names"
        )
        self.assertEqual(len(labels), 4585)
        result = Result(
            np.array([[1, 2, 6, 8]], np.float32),
            np.array([0.9], np.float32),
            np.array([7]),
            [np.ones((6, 5), np.uint8)],
            "roi",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.png"
            save_result(path, image, result, labels)
            self.assertEqual(cv2.imread(str(path)).shape, image.shape)
            bad = Path(tmp) / "bad.names"
            bad.write_text("wrong vocabulary")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_inputs(sample / "test_data/office_desk.jpg", bad)


class TransportTests(unittest.TestCase):
    def test_bad_input_rejected_before_sdk_and_raw_arrays_preserved(self):
        for target in ("x5", "s100"):
            s = resolve_selection(target)
            runtime, data = fake_runtime(s)
            runner = build_runner(
                s,
                runtime_loader=lambda: SimpleNamespace(HB_HBMRuntime=lambda p: runtime),
            )
            task = YOLOE(s, runner=runner)
            a = task.pre_process(np.zeros((320, 640, 3), np.uint8))
            b = task.pre_process(np.zeros((640, 320, 3), np.uint8))
            self.assertEqual(a.context.original_size, (320, 640))
            self.assertEqual(b.context.original_size, (640, 320))
            raw = task.forward(a)
            for name in data:
                self.assertIs(raw[name], data[name])
            calls = runtime.calls
            for bad in ({}, {"m": {}}, {"m": {"wrong": np.zeros(1, np.uint8)}}):
                with self.assertRaises(ValueError):
                    task.forward(bad)
            self.assertEqual(runtime.calls, calls)


class SourceInputMetadataTests(unittest.TestCase):
    def test_source_x5_rgb_nhwc_descriptor_preserves_packed_transport(self):
        selection = resolve_selection("x5")
        runtime, _ = fake_runtime(selection)
        runtime.input_shapes = {"input": (1, 640, 640, 3)}
        runner = build_runner(
            selection,
            runtime_loader=lambda: SimpleNamespace(HB_HBMRuntime=lambda p: runtime),
        )
        task = YOLOE(selection, runner=runner)
        prepared = task.pre_process(np.zeros((320, 640, 3), np.uint8))
        self.assertEqual(prepared.tensors["m"]["input"].shape, (614400,))
        task.forward(prepared)
        self.assertEqual(runtime.calls, 1)


class EmptyResultTests(unittest.TestCase):
    def test_empty_shapes_are_stable_for_all_protocols(self):
        for target, variant in [("x5", "11s"), ("s100", "11s"), ("s100p", "26n")]:
            selection = resolve_selection(target, variant=variant)
            runtime, data = fake_runtime(selection)
            for name, array in data.items():
                if name.startswith("cls_"):
                    array.fill(-30)
            runner = build_runner(
                selection,
                runtime_loader=lambda: SimpleNamespace(HB_HBMRuntime=lambda p: runtime),
            )
            result = YOLOE(selection, runner=runner).predict(
                np.zeros((37, 59, 3), np.uint8)
            )
            self.assertEqual(result.boxes.shape, (0, 4))
            self.assertEqual(result.scores.shape, (0,))
            self.assertEqual(result.class_ids.dtype, np.int64)
            if target == "x5":
                self.assertEqual(result.masks.shape, (0, 37, 59))
            else:
                self.assertEqual(result.masks, [])
