"""Real ONNX structure checks and calibration math; compiler tests use explicit doubles."""

import json
import re
import shlex
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import cv2
from samples.vision.yoloe.conversion.prepare import prepare_conversion
from samples.vision.yoloe.conversion.calibration import calibration_tensor
from samples._shared.yoloe26_geometry import prepare_rgb
from samples.vision.ultralytics_yolo.runtime.python.geometry import (
    resize_with_transform,
)

ROOT = Path(__file__).resolve().parents[4]
LABELS = ROOT / "samples/vision/yoloe/test_data/classes.names"


def graph(path, variant="26n", integer=False):
    import onnx
    from onnx import helper, TensorProto

    nodes = []
    outputs = []
    for stride in (8, 16, 32):
        for kind, channels in [
            ("cls", 4585),
            ("box", 4 if variant.startswith("26") else 64),
            ("mces", 32),
        ]:
            outputs.append(
                (f"{kind}_{stride}", (1, 640 // stride, 640 // stride, channels))
            )
    outputs.append(("protos", (1, 160, 160, 32)))
    for name, shape in outputs:
        dims = helper.make_tensor(name + "_dims", TensorProto.INT64, [4], shape)
        scalar = helper.make_tensor(
            name + "_value",
            TensorProto.INT32 if integer else TensorProto.FLOAT,
            [1],
            [0],
        )
        nodes += [
            helper.make_node("Constant", [], [name + "_shape"], value=dims),
            helper.make_node(
                "ConstantOfShape", [name + "_shape"], [name], value=scalar
            ),
        ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "synthetic-contract-only",
            [
                helper.make_tensor_value_info(
                    "images", TensorProto.FLOAT, [1, 3, 640, 640]
                )
            ],
            [
                helper.make_tensor_value_info(
                    n, TensorProto.INT32 if integer else TensorProto.FLOAT, s
                )
                for n, s in outputs
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


class CalibrationTests(unittest.TestCase):
    def test_x5_raw_vs_s_normalized_and_geometry(self):
        image = np.random.default_rng(8).integers(0, 256, (37, 59, 3), dtype=np.uint8)
        x5 = calibration_tensor(image, "x5", "11s")
        pixels, _ = resize_with_transform(image, (640, 640), 1)
        expected = np.ascontiguousarray(
            pixels[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32
        )
        np.testing.assert_array_equal(x5, expected)
        np.testing.assert_array_equal(
            calibration_tensor(image, "s100", "11s"), expected / 255
        )
        np.testing.assert_array_equal(
            calibration_tensor(image, "s100p", "26n"), prepare_rgb(image)
        )
        with self.assertRaises(ValueError):
            calibration_tensor(image, "s600", "26n")


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for name, value in [("b.png", 140), ("a.png", 33), ("c.png", 240)]:
            cv2.imwrite(str(self.images / name), np.full((37, 59, 3), value, np.uint8))
        self.onnx = self.root / "model.onnx"
        graph(self.onnx)

    def prepare(self, **kwargs):
        options = dict(
            onnx_path=self.onnx,
            names_path=LABELS,
            target="s100",
            variant="26n",
            images=self.images,
            output_dir=self.root / "out",
            sample_count=2,
        )
        options.update(kwargs)
        return prepare_conversion(**options)

    def test_config_only_snapshots_inputs_and_preserves_float_outputs(self):
        report = self.prepare()
        self.assertEqual(report["status"], "config_only")
        self.assertIsNone(report["observed_output_dtype"])
        import yaml

        config = yaml.safe_load((self.root / "out/config.yaml").read_text())
        self.assertNotIn("remove_node_type", config["model_parameters"])
        self.assertNotIn("remove_node_name", config["model_parameters"])
        self.assertEqual(config["model_parameters"]["march"], "nash-e")
        self.assertEqual(config["input_parameters"]["scale_value"], 1 / 255)
        records = json.loads((self.root / "out/calibration.json").read_text())
        self.assertEqual([Path(r["source"]).name for r in records], ["a.png", "c.png"])
        tensor = np.load(
            self.root / "out/calibration" / records[0]["tensor"], allow_pickle=False
        )
        self.assertEqual(tensor.dtype, np.float32)
        self.assertLessEqual(float(tensor.max()), 1)
        self.assertTrue((self.root / "out/source/model.onnx").is_file())
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_x5_raw_file_and_source_softmax_override(self):
        graph(self.onnx, "11s")
        report = self.prepare(target="x5", variant="11s")
        import yaml

        config = yaml.safe_load((self.root / "out/config.yaml").read_text())
        self.assertEqual(config["model_parameters"]["march"], "bayes-e")
        self.assertEqual(config["input_parameters"]["norm_type"], "data_scale")
        rows = json.loads((self.root / "out/calibration.json").read_text())
        tensor = np.fromfile(
            self.root / "out/calibration" / rows[0]["tensor"], dtype=np.float32
        ).reshape(1, 3, 640, 640)
        self.assertEqual(float(tensor.max()), 127)
        self.assertIn("source attention node absent", report["warnings"])

    def test_invalid_graph_vocab_target_rejected_without_output(self):
        for kwargs in [
            dict(target="s600"),
            dict(variant="11s"),
            dict(sample_count=0),
            dict(compile_model=True, compiler=str(self.root / "missing-compiler")),
        ]:
            with self.assertRaises(ValueError):
                self.prepare(**kwargs)
            self.assertFalse((self.root / "out").exists())
        names = self.root / "bad.names"
        names.write_text("wrong")
        with self.assertRaises(ValueError):
            self.prepare(names_path=names)
        graph(self.onnx, integer=True)
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse((self.root / "out").exists())

    def test_fake_compiler_failure_and_success_are_not_precision_verification(self):
        compiler = self.root / "compiler"
        compiler.write_text(
            f'#!{sys.executable}\nimport sys\nprint("intentional compiler failure")\nsys.exit(7)\n'
        )
        compiler.chmod(0o755)
        result = self.prepare(compile_model=True, compiler=str(compiler))
        self.assertEqual(result["status"], "compile_failed")
        self.assertEqual(result["compiler"]["returncode"], 7)
        self.assertIn(
            "intentional compiler failure", (self.root / "out/compile.log").read_text()
        )
        compiler.write_text(
            f'#!{sys.executable}\nfrom pathlib import Path\nimport sys,yaml\nc=yaml.safe_load(Path(sys.argv[2]).read_text())["model_parameters"]\np=Path(c["working_dir"]);p.mkdir(parents=True)\n(p/(c["output_model_file_prefix"]+".hbm")).write_bytes(b"synthetic-not-a-model")\n'
        )
        result = self.prepare(
            output_dir=self.root / "second", compile_model=True, compiler=str(compiler)
        )
        self.assertEqual(result["status"], "compiled_unverified")
        self.assertIsNone(result["observed_output_dtype"])
        self.assertEqual(result["board"], "not-run")
        self.assertEqual(len(result["artifact"]["sha256"]), 64)

    def test_bad_proto_and_unreadable_image_leave_honest_status(self):
        self.onnx.write_bytes(b"not-an-onnx-protobuf")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse((self.root / "out").exists())
        graph(self.onnx)
        (self.images / "a.png").write_bytes(b"broken image")
        with self.assertRaisesRegex(ValueError, "Unreadable"):
            self.prepare()
        status = json.loads((self.root / "out/conversion.json").read_text())
        self.assertEqual(status["status"], "preparation_failed")
        self.assertIsNone(status["compiler"])

    def test_all_fourteen_target_variants_produce_reviewable_configs(self):
        from samples.vision.yoloe.runtime.python.model_binding import list_models
        from samples.vision.yoloe.conversion.configuration import MARCHES
        import yaml

        for target, variant, asset in list_models():
            graph(self.onnx, variant)
            directory = self.root / (target + "-" + variant)
            status = self.prepare(
                target=target, variant=variant, output_dir=directory, sample_count=1
            )
            config = yaml.safe_load((directory / "config.yaml").read_text())
            self.assertEqual(config["model_parameters"]["march"], MARCHES[target])
            self.assertEqual(status["source_asset_id"], asset.reference)
            self.assertEqual(status["status"], "config_only")
            self.assertIsNone(status["artifact"])
            self.assertFalse(
                any("remove_node" in name for name in config["model_parameters"])
            )

    def test_external_tensor_and_nonstatic_shapes_are_rejected(self):
        import onnx

        model = onnx.load(self.onnx)
        tensor = model.graph.node[1].attribute[0].t
        tensor.data_location = onnx.TensorProto.EXTERNAL
        tensor.external_data.add(key="location", value="untracked-weight.bin")
        self.onnx.write_bytes(model.SerializeToString())
        with self.assertRaisesRegex(ValueError, "External"):
            self.prepare()
        graph(self.onnx)
        model = onnx.load(self.onnx)
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
        self.onnx.write_bytes(model.SerializeToString())
        with self.assertRaises(ValueError):
            self.prepare()

    def test_documented_preparation_commands_run_without_compiler(self):
        blocks = []
        for language in ("README.md", "README_cn.md"):
            page = ROOT / "samples/vision/yoloe/conversion" / language
            candidates = [
                block
                for block in re.findall(r"```bash\n(.*?)```", page.read_text(), re.S)
                if "--sample-count 100" in block
            ]
            self.assertEqual(len(candidates), 1)
            blocks.append(candidates[0])
            command = "\n".join(
                line for line in candidates[0].splitlines() if not line.startswith("#")
            )
            argv = shlex.split(command.replace("\\\n", " "))
            replacements = {
                "python3": sys.executable,
                "/work/export26n/yoloe_26n_seg_pf.onnx": str(self.onnx),
                "/work/export26n/yoloe_26n_seg_pf.names": str(LABELS),
                "/work/calibration-images": str(self.images),
                "/work/yoloe26n-s100-config": str(self.root / language),
            }
            result = subprocess.run(
                [replacements.get(token, token) for token in argv],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "config_only")
            self.assertIsNone(report["compiler"])
            self.assertEqual(report["calibration"]["actual"], 3)
        self.assertEqual(blocks[0], blocks[1])
