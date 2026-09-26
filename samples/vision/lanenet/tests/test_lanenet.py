"""LaneNet raw embedding/label contracts, never a clustering claim."""

from pathlib import Path
from types import SimpleNamespace
import ast
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from samples.vision.lanenet.runtime.python.model_binding import (
    bind_model,
    resolve_selection,
    list_available_assets,
)
from samples.vision.lanenet.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.lanenet.runtime.python.lanenet import LaneNetTask
from samples.vision.lanenet.runtime.python.visualization import (
    embedding_image,
    binary_image,
)


def metadata(extra=False):
    outputs = {
        "instance_seg_logits": ((1, 3, 256, 512), "float32"),
        "binary_seg_pred": ((1, 1, 256, 512), "S64"),
    }
    if extra:
        outputs["observed_aux"] = ((1, 2, 256, 512), "float32")
    return {
        "model_name": "lane",
        "model_names": ["lane"],
        "input_names": ["input"],
        "input_shapes": {"input": (1, 3, 256, 512)},
        "input_dtypes": {"input": "float32"},
        "output_names": list(reversed(outputs)),
        "output_shapes": {k: v[0] for k, v in outputs.items()},
        "output_dtypes": {k: v[1] for k, v in outputs.items()},
    }


def raw_outputs(extra=False):
    result = {
        "instance_seg_logits": np.linspace(
            -0.2, 1.2, 3 * 256 * 512, dtype=np.float32
        ).reshape(1, 3, 256, 512),
        "binary_seg_pred": np.indices((1, 1, 256, 512))[-1] % 2,
    }
    if extra:
        result["observed_aux"] = np.zeros((1, 2, 256, 512), np.float32)
    return result


class LaneTests(unittest.TestCase):
    def task(self, extra=False):
        binding = bind_model(resolve_selection("s100"), metadata(extra))
        return LaneNetTask(lambda inputs: raw_outputs(extra), binding)

    def test_explicit_download_uses_exact_manifest_and_target_subdirectory(self):
        import contextlib
        import io
        from samples.vision.lanenet.model import download

        with patch.object(
            download, "download_asset", return_value="a" * 64
        ) as call, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                download.main(
                    ["--target", "s100", "--output-dir", "/tmp/lane-prepared"]
                ),
                0,
            )
        asset, path = call.call_args.args
        self.assertEqual(asset.reference, "s:lanenet:s100/lanenet256x512.hbm")
        self.assertEqual(path, Path("/tmp/lane-prepared/s100/lanenet256x512.hbm"))

    def test_only_exact_s100_asset(self):
        self.assertEqual(len(list_available_assets()), 1)
        self.assertEqual(
            resolve_selection("auto").asset.reference,
            "s:lanenet:s100/lanenet256x512.hbm",
        )
        for target in ("x5", "s100p", "s600"):
            with self.assertRaises(ValueError):
                resolve_selection(target)
        with self.assertRaises(ValueError):
            resolve_selection("s100", model_path="arbitrary.hbm")

    def test_role_names_and_auxiliary_metadata_not_output_indices(self):
        b = bind_model(resolve_selection("s100"), metadata(True))
        self.assertEqual(b.embedding_name, "instance_seg_logits")
        self.assertEqual(b.binary_name, "binary_seg_pred")
        self.assertEqual(len(b.metadata.output_names), 3)
        for field, value in [
            ("output_names", ["binary_seg_pred"]),
            ("input_shapes", {"input": (1, 3, 512, 256)}),
            (
                "output_dtypes",
                {"instance_seg_logits": "int32", "binary_seg_pred": "int64"},
            ),
        ]:
            m = metadata()
            m[field] = value
            with self.assertRaises(ValueError):
                bind_model(resolve_selection("s100"), m)

    def test_source_preprocess_exact(self):
        image = np.random.default_rng(5).integers(0, 256, (37, 71, 3), np.uint8)
        source = (
            cv2.resize(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                (512, 256),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            / 255
        )
        source = (
            source.transpose(2, 0, 1)
            - np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
        ) / np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
        result = self.task().pre_process(image)
        np.testing.assert_array_equal(result["input"], source[None])
        native_order = cv2.cvtColor(
            cv2.resize(image, (512, 256), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        np.testing.assert_array_equal(
            native_order,
            cv2.resize(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                (512, 256),
                interpolation=cv2.INTER_AREA,
            ),
        )

    def test_postprocess_keeps_raw_embedding_and_discrete_model_grid(self):
        raw = raw_outputs(True)
        result = self.task(True).post_process(raw)
        np.testing.assert_array_equal(result.embedding, raw["instance_seg_logits"][0])
        self.assertFalse(np.shares_memory(result.embedding, raw["instance_seg_logits"]))
        self.assertEqual(result.binary.shape, (256, 512))
        self.assertEqual(result.binary.dtype, np.uint8)
        self.assertEqual(set(np.unique(result.binary)), {0, 1})
        self.assertLess(float(result.embedding.min()), 0)
        self.assertGreater(float(result.embedding.max()), 1)

    def test_bad_labels_and_nonfinite_or_wrong_shape_fail(self):
        t = self.task()
        for bad in (2, -1):
            raw = raw_outputs()
            raw["binary_seg_pred"][0, 0, 0, 0] = bad
            with self.assertRaises(ValueError):
                t.post_process(raw)
        for value in (
            np.zeros((3, 256, 512), np.float32),
            np.full((1, 3, 256, 512), np.nan, np.float32),
        ):
            raw = raw_outputs()
            raw["instance_seg_logits"] = value
            with self.assertRaises(ValueError):
                t.post_process(raw)
        with self.assertRaises(ValueError):
            t.pre_process(np.zeros((0, 3, 3), np.uint8))

    def test_visualization_clip_round_separate_from_raw(self):
        x = np.tile(np.array([[-0.1, 0.1, 0.5, 1, 1.2]], np.float32), (3, 1, 1))
        actual = embedding_image(x)
        np.testing.assert_array_equal(actual[:, :, 0], [[0, 26, 128, 255, 255]])
        np.testing.assert_array_equal(
            binary_image(np.array([[0, 1]], np.uint8)), [[0, 255]]
        )

    def test_runner_preserves_aux_and_int64_owned_outputs(self):
        m = metadata(True)
        raw = raw_outputs(True)
        runtime = SimpleNamespace(
            **{k: v for k, v in m.items() if k != "model_name"},
            run=lambda inputs: {"lane": raw}
        )
        runner = RuntimeModelRunner(resolve_selection("s100"), runtime=runtime)
        t = LaneNetTask(runner, runner.load())
        result = t.forward(t.pre_process(np.zeros((8, 9, 3), np.uint8)))
        self.assertEqual(set(result), set(raw))
        self.assertEqual(result["binary_seg_pred"].dtype, np.int64)
        for name in result:
            self.assertFalse(np.shares_memory(raw[name], result[name]))

    def test_stage_purity_and_real_gate(self):
        p = Path(__file__).resolve().parents[1] / "runtime/python/lanenet.py"
        tree = ast.parse(p.read_text())
        c = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "LaneNetTask"
        )
        self.assertEqual(
            {n.name for n in c.body if isinstance(n, ast.FunctionDef)},
            {"__init__", "pre_process", "forward", "post_process", "predict"},
        )
        with patch(
            "samples.vision.lanenet.runtime.python.model_runner.require_execution_target",
            side_effect=ValueError("wrong board"),
        ):
            with self.assertRaisesRegex(ValueError, "wrong board"):
                RuntimeModelRunner(resolve_selection("s100")).load()


if __name__ == "__main__":
    unittest.main()
