"""Planning contract tests with actual bundled float features; no actuation or SDK."""

import ast
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import numpy as np
from samples.vision.diffusiondrive.runtime.python.model_binding import (
    bind_model,
    resolve_selection,
    INPUT_SHAPES,
    OUTPUT_SHAPES,
)
from samples.vision.diffusiondrive.runtime.python.diffusiondrive import (
    DiffusionDriveTask,
)
from samples.vision.diffusiondrive.runtime.python.quantization import quantize, decode

SOURCE = (
    Path(__file__).resolve().parents[4] / "platforms/s/samples/vision/diffusiondrive"
)


def metadata(integer=False):
    def q():
        return SimpleNamespace(
            quant_type="SCALE",
            scale=np.array([0.02], np.float32),
            zero_point=np.array([0]),
            axis=0,
        )

    return dict(
        model_name="plan",
        model_names=["plan"],
        input_names=list(reversed(INPUT_SHAPES)),
        output_names=list(reversed(OUTPUT_SHAPES)),
        input_shapes=INPUT_SHAPES,
        output_shapes=OUTPUT_SHAPES,
        input_dtypes={n: "S16" if integer else "F32" for n in INPUT_SHAPES},
        output_dtypes={n: "S16" if integer else "F32" for n in OUTPUT_SHAPES},
        input_quants={n: q() for n in INPUT_SHAPES} if integer else {},
        output_quants={n: q() for n in OUTPUT_SHAPES} if integer else {},
    )


def arrays(name):
    with np.load(SOURCE / "test_data" / name, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


class DiffusionTests(unittest.TestCase):
    def test_targets_and_exact_asset_binding(self):
        for target in ("s100p", "s600"):
            s = resolve_selection(target)
            self.assertEqual(s.asset.sha256 is not None, True)
            self.assertIn(target, s.asset.reference)
        for target in ("s100", "x5"):
            with self.assertRaises(ValueError):
                resolve_selection(target)
        with patch(
            "samples.vision.diffusiondrive.runtime.python.model_binding.resolve_target",
            side_effect=ValueError("unknown"),
        ):
            with self.assertRaises(ValueError):
                resolve_selection("auto")
        with self.assertRaises(ValueError):
            resolve_selection("s600", model_path="/tmp/a.hbm")
        with self.assertRaises(ValueError):
            resolve_selection(
                "s600", asset_id=resolve_selection("s100p").asset.reference
            )

    def test_float_stages_preserve_source_semantics_and_own_results(self):
        features = arrays("reference_inputs.npz")
        raw = arrays("reference_outputs.npz")
        binding = bind_model(resolve_selection("s600"), metadata())
        task = DiffusionDriveTask(lambda x: raw, binding)
        prepared = task.pre_process(features)
        for name in features:
            np.testing.assert_array_equal(prepared[name], features[name])
        result = task.predict(features)
        np.testing.assert_array_equal(result["trajectory"], raw["trajectory"])
        scores = 1 / (1 + np.exp(-np.clip(raw["agent_labels"], -60, 60)))
        np.testing.assert_array_equal(result["agent_scores"], scores)
        np.testing.assert_array_equal(result["agent_mask"], scores >= 0.5)
        np.testing.assert_array_equal(
            result["bev_labels"],
            np.argmax(raw["bev_semantic_map"], axis=1).astype(np.uint8),
        )
        result["trajectory"].fill(0)
        self.assertTrue(raw["trajectory"].any())
        prepared["camera"].fill(0)
        self.assertTrue(features["camera"].any())

    def test_quantized_preprocess_matches_source_on_packaged_features(self):
        m = metadata(True)
        binding = bind_model(resolve_selection("s100p"), m)
        features = arrays("reference_inputs.npz")
        actual = DiffusionDriveTask(lambda _: None, binding).pre_process(features)
        for name, value in features.items():
            expected = np.clip(
                np.rint(value / float(np.float32(0.02))), -32768, 32767
            ).astype(np.int16)
            np.testing.assert_array_equal(actual[name], expected)

    def test_per_axis_output_scalar_zero_and_invalid_quantization(self):
        m = metadata(True)
        m["output_quants"]["trajectory"] = SimpleNamespace(
            quant_type="SCALE",
            scale=np.array([0.25, 0.5, 1], np.float32),
            zero_point=np.array([2]),
            axis=-1,
        )
        b = bind_model(resolve_selection("s600"), m)
        raw = np.arange(24, dtype=np.int16).reshape(1, 8, 3)
        expected = (raw.astype(np.float32) - 2) * np.array([0.25, 0.5, 1], np.float32)
        np.testing.assert_array_equal(
            decode(raw, b.output_transforms["trajectory"]), expected
        )
        for scale in (-1, 0, float("nan")):
            bad = metadata(True)
            bad["input_quants"]["status"].scale = np.array([scale])
            with self.assertRaises(ValueError):
                bind_model(resolve_selection("s600"), bad)
        bad = metadata(True)
        bad["input_quants"]["status"].scale = np.ones(8)
        with self.assertRaises(ValueError):
            bind_model(resolve_selection("s600"), bad)

    def test_metadata_rejects_wrong_shapes_names_types_or_missing_quant(self):
        for modify in (
            lambda m: m.update(input_names=["camera"] * 4),
            lambda m: m.update(output_shapes={**OUTPUT_SHAPES, "trajectory": (1, 24)}),
            lambda m: m.update(input_quants={}),
            lambda m: m.update(
                output_dtypes={**m["output_dtypes"], "trajectory": "S64"}
            ),
        ):
            m = metadata(True)
            modify(m)
            with self.assertRaises(ValueError):
                bind_model(resolve_selection("s600"), m)

    def test_features_and_outputs_fail_before_plausible_result(self):
        b = bind_model(resolve_selection("s600"), metadata())
        task = DiffusionDriveTask(lambda _: None, b)
        f = arrays("reference_inputs.npz")
        for invalid in (
            {**f, "extra": np.zeros(1)},
            {**f, "status": np.zeros((8,), np.float32)},
            {**f, "status": np.full((1, 8), np.nan, np.float32)},
        ):
            with self.assertRaises(ValueError):
                task.pre_process(invalid)
        raw = arrays("reference_outputs.npz")
        raw["trajectory"][0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            task.post_process(raw)
        for threshold in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                DiffusionDriveTask(lambda _: None, b, threshold)

    def test_integer_saturation_does_not_wrap_at_uint32_boundary(self):
        from samples.vision.diffusiondrive.runtime.python.quantization import transform

        q = SimpleNamespace(
            quant_type="SCALE", scale=np.array([1.0]), zero_point=np.array([0]), axis=0
        )
        spec = transform("uint32", (1, 2), q, input_tensor=True)
        actual = quantize(np.array([[-1, 2**32]], np.float32), spec)
        np.testing.assert_array_equal(actual, np.array([[0, 2**32 - 1]], np.uint32))

    def test_real_named_runner_preserves_four_inputs_raw_outputs_and_scheduling(self):
        from samples.vision.diffusiondrive.runtime.python.model_runner import (
            RuntimeModelRunner,
        )

        m = metadata(True)
        features = arrays("reference_inputs.npz")
        raw = {n: np.zeros(shape, np.int16) for n, shape in OUTPUT_SHAPES.items()}
        seen = []
        schedules = []
        runtime = SimpleNamespace(
            **{k: v for k, v in m.items() if k != "model_name"},
            run=lambda values: (seen.append(values) or {"plan": raw}),
            set_scheduling_params=lambda **kw: schedules.append(kw)
        )
        runner = RuntimeModelRunner(resolve_selection("s600"), runtime=runtime)
        binding = runner.load()
        runner.set_scheduling_params(priority=0, bpu_cores=[0])
        task = DiffusionDriveTask(runner, binding)
        result = task.predict(features)
        self.assertEqual(set(seen[0]["plan"]), set(INPUT_SHAPES))
        self.assertTrue(all(a.dtype == np.int16 for a in seen[0]["plan"].values()))
        self.assertEqual(
            schedules, [{"priority": {"plan": 0}, "bpu_cores": {"plan": [0]}}]
        )
        self.assertTrue(result["agent_mask"].all())
        owned = task.forward(task.pre_process(features))
        raw["trajectory"].fill(100)
        self.assertFalse(owned["trajectory"].any())
        m["input_quants"]["status"].scale[0] = 99
        self.assertAlmostEqual(binding.input_transforms["status"].scale[0], 0.02)

    def test_real_path_gates_identity_before_sdk_construction(self):
        from samples.vision.diffusiondrive.runtime.python.model_runner import (
            RuntimeModelRunner,
        )

        with patch(
            "samples.vision.diffusiondrive.runtime.python.model_runner.require_execution_target",
            side_effect=ValueError("wrong board"),
        ) as gate, patch(
            "samples._shared.single_array_runner._default_runtime_factory"
        ) as factory:
            with self.assertRaises(ValueError):
                RuntimeModelRunner(resolve_selection("s100p")).load()
            gate.assert_called_once_with("s100p")
            factory.assert_not_called()

    def test_download_uses_target_filename_and_published_hash(self):
        from samples.vision.diffusiondrive.model import download
        import contextlib, io

        with patch.object(
            download, "download_asset", return_value="fixture"
        ) as call, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                download.main(["--target", "s100p", "--output-dir", "/tmp/planning"]), 0
            )
        asset, path = call.call_args.args
        self.assertEqual(len(asset.sha256), 64)
        self.assertEqual(
            path, Path("/tmp/planning/s100p/diffusiondrive_r34_256x1024_s100p.hbm")
        )

    def test_all_six_packaged_cases_match_actual_source_postprocess_and_render(self):
        from samples.vision.diffusiondrive.runtime.python.visualization import (
            render_result,
        )
        import cv2, tempfile

        spec = importlib.util.spec_from_file_location(
            "diffusion_source_fixture", SOURCE / "runtime/python/diffusiondrive.py"
        )
        legacy = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "hbm_runtime": ModuleType("hbm_runtime"),
                "diffusion_source_fixture": legacy,
            },
        ):
            spec.loader.exec_module(legacy)
        source = legacy.DiffusionDrive.__new__(legacy.DiffusionDrive)
        source.cfg = legacy.DiffusionDriveConfig("not-loaded", 0.5)
        source.model_name = "plan"
        source.model = SimpleNamespace(
            output_quants={
                "plan": {
                    n: SimpleNamespace(
                        scale=np.array([]), zero_point=np.array([]), axis=0
                    )
                    for n in OUTPUT_SHAPES
                }
            }
        )
        binding = bind_model(resolve_selection("s600"), metadata())
        task = DiffusionDriveTask(lambda _: None, binding)
        cases = [("reference_inputs.npz", "reference_outputs.npz")] + [
            (
                str(p.relative_to(SOURCE / "test_data") / "inputs.npz"),
                str(p.relative_to(SOURCE / "test_data") / "reference_outputs.npz"),
            )
            for p in sorted((SOURCE / "test_data").glob("case_*"))
        ]
        self.assertEqual(len(cases), 6)
        for inp, out in cases:
            features = arrays(inp)
            raw = arrays(out)
            old = source.post_process({"plan": raw})
            new = task.post_process(raw)
            self.assertEqual(set(old), set(new))
            for name in old:
                np.testing.assert_array_equal(
                    new[name], old[name], err_msg=out + ":" + name
                )
            actual = render_result(features, new, platform_name="S600")
            with tempfile.TemporaryDirectory() as tmp:
                image_path = str(Path(tmp) / "source.png")
                legacy.render_result(features, old, image_path, platform_name="S600")
                np.testing.assert_array_equal(actual, cv2.imread(image_path))

    def test_float_none_descriptor_with_zero_placeholder_is_pass_through(self):
        m = metadata()
        empty = SimpleNamespace(
            quant_type="NONE",
            scale=np.array([], np.float32),
            zero_point=np.array([0], np.int32),
            axis=0,
        )
        m["input_quants"] = {name: empty for name in INPUT_SHAPES}
        m["output_quants"] = {name: empty for name in OUTPUT_SHAPES}
        binding = bind_model(resolve_selection("s600"), m)
        features = arrays("reference_inputs.npz")
        raw = arrays("reference_outputs.npz")
        task = DiffusionDriveTask(lambda _: raw, binding)
        for name, value in task.pre_process(features).items():
            np.testing.assert_array_equal(value, features[name])
        np.testing.assert_array_equal(
            task.predict(features)["trajectory"], raw["trajectory"]
        )

    def test_task_has_only_three_stages_and_predict(self):
        p = Path(__file__).resolve().parents[1] / "runtime/python/diffusiondrive.py"
        tree = ast.parse(p.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DiffusionDriveTask"
        )
        self.assertEqual(
            {n.name for n in cls.body if isinstance(n, ast.FunctionDef)},
            {"__init__", "pre_process", "forward", "post_process", "predict"},
        )


if __name__ == "__main__":
    unittest.main()
