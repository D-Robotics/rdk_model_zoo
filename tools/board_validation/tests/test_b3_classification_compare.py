"""Host fake-runtime behavior tests for the B3 classification board tool.

Every test drives the real tool protocol (selection -> gate -> fixed-source
load -> unified entry -> evidence) against an injected fake SDK runtime, so
``hbm_runtime`` is never required on the host.  Nothing here is a board claim.

Run with:
  python3 -m unittest discover -s tools/board_validation/tests -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

TESTS_DIR = Path(__file__).resolve().parent


def load_tool():
    """Import the tool by path with proper sys.modules registration."""

    module_name = "b3_classification_compare"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        module_name, TESTS_DIR.parent / "b3_classification_compare.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tool = load_tool()

_DEFAULT_ASSET = {
    "convnext": "x5:convnext:ConvNeXt_atto_224x224_nv12.bin",
    "edgenext": "x5:edgenext:EdgeNeXt_base_224x224_nv12.bin",
    "fasternet": "x5:fasternet:FasterNet_S_224x224_nv12.bin",
    "fastvit": "x5:fastvit:FastViT_S12_224x224_nv12.bin",
}

_DTYPE_MAP = {"F32": np.float32, "F64": np.float64, "F16": np.float16}


class FakeRuntime:
    """Deterministic X5-shaped SDK stand-in: one model, one input, one output."""

    def __init__(self, sample, logits, *, output_shape=(1, 1000, 1, 1),
                 output_dtype="F32", quant=None, run_error=None):
        self.model_names = [sample]
        self.input_names = {sample: ["data"]}
        self.input_shapes = {sample: {"data": (1, 3, 224, 224)}}
        self.input_dtypes = {sample: {"data": "U8"}}
        self.output_names = {sample: ["prob"]}
        self.output_shapes = {sample: {"prob": output_shape}}
        self.output_dtypes = {sample: {"prob": output_dtype}}
        self.output_quants = {sample: {"prob": quant}} if quant is not None else {}
        self.logits = np.asarray(logits)
        self.run_error = run_error
        self.runs = []
        self.schedules = []

    def run(self, inputs):
        self.runs.append(inputs)
        if self.run_error is not None:
            raise self.run_error
        array = (
            self.logits.reshape(self.output_shapes[self.model_names[0]]["prob"])
            .astype(_DTYPE_MAP[self.output_dtypes[self.model_names[0]]["prob"]])
        )
        return {self.model_names[0]: {"prob": array}}

    def set_scheduling_params(self, **kwargs):
        self.schedules.append(kwargs)


class _BoardQuantParams:
    """Mimics hbm_runtime.QuantParams: attributes read, any copy refuses."""

    def __init__(self):
        self.quant_type = types.SimpleNamespace(name="SCALE")
        self.scale = np.asarray(0.25, dtype=np.float32)
        self.zero_point = np.asarray(7, dtype=np.int32)
        self.axis = 3

    def __deepcopy__(self, memo):
        raise TypeError("cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object")

    def __copy__(self):
        raise TypeError("cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object")


def _base_logits():
    """Distinct, well-separated values: no float32 ties, no exp underflow."""

    return np.linspace(-20.0, 20.0, 1000).astype(np.float32)


class _Harness:
    """Reusable CLI driver: fixture assets + injected fake SDK factory."""

    def __init__(self, tmp: Path, sample: str):
        self.tmp = tmp
        self.sample = sample
        self.model = tmp / "model.fixture"
        self.model.write_bytes(b"b3-fixture-model-bytes")
        self.labels = tmp / "labels.txt"
        self.labels.write_text("".join(f"class-{i}\n" for i in range(1000)))
        self.image_path = tmp / "input.png"
        image = (np.arange(37 * 61 * 3) % 256).astype(np.uint8).reshape(37, 61, 3)
        if not cv2.imwrite(str(self.image_path), image):
            raise OSError("could not write test image fixture")
        self.fakes = []

    def configure(self, index):
        """Per-side fake policy; index 0 is the legacy side, 1 the unified side."""

        return dict(logits=_base_logits().copy())

    def create(self, path):
        config = self.configure(len(self.fakes))
        fake = FakeRuntime(self.sample, config.pop("logits"), **config)
        self.fakes.append(fake)
        return fake

    def argv(self, **overrides):
        args = [
            "--sample", self.sample,
            "--target", "x5",
            "--output-dir", str(self.tmp / "evidence"),
            "--model-path", str(self.model),
            "--asset-id", _DEFAULT_ASSET[self.sample],
            "--test-img", str(self.image_path),
            "--label-file", str(self.labels),
        ]
        for key, value in overrides.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, (list, tuple)):
                args += [flag, *[str(item) for item in value]]
            else:
                args += [flag, str(value)]
        return args

    def run(self, **overrides):
        with patch.object(tool, "require_execution_target", return_value="x5"), \
                patch(
                    "samples._shared.model_runner._default_runtime_factory",
                    return_value=self.create,
                ), contextlib.redirect_stderr(io.StringIO()) as err, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = tool.main(self.argv(**overrides))
        return rc, err.getvalue(), out.getvalue()

    def evidence(self):
        return json.loads((self.tmp / "evidence" / "comparison.json").read_text())


class B3CompareProtocolTests(unittest.TestCase):
    def _run_sample(self, tmp, sample):
        harness = _Harness(tmp, sample)
        rc, err, _ = harness.run()
        return rc, err, harness

    def test_full_protocol_passes_for_all_four_samples(self):
        """Both entries execute separately and agree byte-exactly on inputs."""

        for sample in sorted(_DEFAULT_ASSET):
            with self.subTest(sample=sample), tempfile.TemporaryDirectory() as td:
                rc, err, harness = self._run_sample(Path(td), sample)
                self.assertEqual(rc, 0, err or harness.evidence().get("error"))
                stored = harness.evidence()
                self.assertTrue(stored["passed"])
                self.assertTrue(all(stored["comparison"]["checks"].values()))
                self.assertEqual(stored["return_code"], 0)
                self.assertEqual(stored["source_ref"], tool.SOURCE_REF)
                self.assertEqual(stored["variant"], {
                    "convnext": "atto", "edgenext": "base",
                    "fasternet": "s", "fastvit": "s12"}[sample])
                # Two separate SDK instances, one run each: no side substituted.
                self.assertEqual(len(harness.fakes), 2)
                self.assertEqual([len(fake.runs) for fake in harness.fakes], [1, 1])
                for fake in harness.fakes:
                    self.assertEqual(
                        fake.schedules,
                        [{"priority": {sample: 0}, "bpu_cores": {sample: [0]}}],
                    )
                checks = stored["comparison"]["checks"]
                self.assertTrue(checks["inputs_exact_bytes"])
                self.assertTrue(checks["inputs_uint8_finite"])
                self.assertTrue(checks["raw_shape_equal"])
                self.assertTrue(checks["raw_dtype_equal"])
                self.assertTrue(checks["raw_finite"])
                self.assertTrue(checks["topk_counts_match_request"])
                self.assertTrue(checks["topk_ids_unique"])
                self.assertTrue(checks["topk_ids_equal"])
                self.assertTrue(checks["topk_scores_within_tolerance"])
                self.assertTrue(checks["topk_labels_equal"])
                self.assertFalse(stored["comparison"]["tie"]["detected"])
                raw = stored["comparison"]["raw_outputs"]["prob"]
                self.assertEqual(raw["max_abs_diff"], 0.0)
                self.assertEqual(raw["nonzero_diff_count"], 0)
                # Legacy pre input keeps its (1, 3H/2, W, 1) view; unified is
                # the flat 1-D buffer; the bytes must still match exactly.
                inputs = stored["comparison"]["inputs"]["data"]
                self.assertEqual(inputs["legacy_shape"], [1, 336, 224, 1])
                self.assertEqual(inputs["unified_shape"], [75264])
                self.assertEqual(inputs["legacy_bytes"], inputs["unified_bytes"])
                self.assertEqual(
                    stored["comparison"]["topk"]["legacy_labels"],
                    stored["comparison"]["topk"]["unified_labels"],
                )
                # Identity and dependency evidence.
                self.assertEqual(stored["board"]["resolved_target"], "x5")
                self.assertIsNotNone(stored["git"]["head"])
                self.assertIn(f"platforms/x5/samples/vision/{sample}/runtime/python/{sample}.py",
                              stored["code_sha256"])
                # Source closure is really verified against the pin's blobs,
                # not just declared via the source_ref constant.
                closure = stored["source_closure"]
                self.assertTrue(closure["verified"], closure)
                self.assertEqual(len(closure["files"]), 5)  # entry + 4 deps
                self.assertTrue(all(entry["pin_sha256"] for entry in closure["files"]))
                # The executed dependencies come from the pin-verified
                # platforms/x5 snapshot, not the drifted root utils/ copies.
                for dep in ("utils.py_utils.file_io", "utils.py_utils.preprocess"):
                    resolved = stored["legacy_dependency_resolution"][dep]
                    self.assertIsNotNone(resolved["observed_file"])
                    self.assertIn("platforms/x5/utils/py_utils", resolved["observed_file"])
                    # Not the drifted root utils/ copy of the same module.
                    self.assertNotEqual(
                        resolved["observed_file"],
                        str(tool._ROOT / "utils" / "py_utils"
                            / resolved["pin_path"].split("/")[-1]),
                    )
                    self.assertIsNotNone(resolved["observed_sha256"])
                    self.assertIsNotNone(resolved["pin_sha256"])
                    self.assertTrue(resolved["matches_pin"])
                # Isolated install: no utils.* module leaks into sys.modules.
                self.assertFalse(
                    [name for name in sys.modules if name == "utils" or name.startswith("utils.")]
                )
                self.assertIn("legacy", stored["metadata"])
                self.assertIn("unified", stored["metadata"])
                self.assertEqual(stored["artifacts"]["model"]["publisher_sha256"], None)
                self.assertEqual(
                    stored["artifacts"]["model"]["observed_sha256"],
                    tool.sha256_file(harness.model),
                )
                files = sorted((Path(td) / "evidence").glob("*.npy"))
                self.assertEqual(len(files), 13, [p.name for p in files])
                self.assertEqual(len(stored["arrays"]), 13)
                self.assertIsNotNone(stored["finished_utc"])
                self.assertTrue(stored["argv"])
                self.assertTrue(stored["cwd"])

    def test_variant_and_asset_exact_selection(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            harness = _Harness(tmp, "edgenext")
            rc, _, _ = harness.run(
                variant="small",
                asset_id="x5:edgenext:EdgeNeXt_small_224x224_nv12.bin",
            )
            self.assertEqual(rc, 0)
            self.assertEqual(harness.evidence()["asset_id"],
                             "x5:edgenext:EdgeNeXt_small_224x224_nv12.bin")
            self.assertEqual(harness.evidence()["variant"], "small")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            harness = _Harness(tmp, "fastvit")
            rc, _, _ = harness.run(asset_id="x5:fastvit:FastViT_SA12_224x224_nv12.bin")
            self.assertEqual(rc, 0)
            self.assertEqual(harness.evidence()["variant"], "sa12")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            harness = _Harness(tmp, "fasternet")
            rc, _, _ = harness.run(
                variant="t0",
                asset_id="x5:fasternet:FasterNet_T0_224x224_nv12.bin",
            )
            self.assertEqual(rc, 0)
            self.assertEqual(harness.evidence()["asset_id"],
                             "x5:fasternet:FasterNet_T0_224x224_nv12.bin")

    def test_unknown_variant_and_model_path_without_asset_id_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            harness = _Harness(tmp, "convnext")
            rc, err, _ = harness.run(variant="nano")
            self.assertEqual(rc, 2)
            self.assertIn("Unknown sample variant", err)
            self.assertFalse((tmp / "evidence").exists())
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            harness = _Harness(tmp, "convnext")
            argv = harness.argv()
            argv.remove("--asset-id")
            argv.remove(_DEFAULT_ASSET["convnext"])
            with patch.object(tool, "require_execution_target", return_value="x5"), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                rc = tool.main(argv)
            self.assertEqual(rc, 2)
            self.assertIn("requires --asset-id", err.getvalue())

    def test_wrong_target_is_a_precise_error_not_a_default(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            harness = _Harness(tmp, "convnext")
            argv = [value if value != "x5" else "s100" for value in harness.argv()]
            with patch.object(tool, "require_execution_target", return_value="x5"), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                rc = tool.main(argv)
            self.assertEqual(rc, 2)
            self.assertIn("No published sample asset matches target='s100'", err.getvalue())
            self.assertFalse((tmp / "evidence").exists())

    def test_missing_model_keeps_error_evidence_record(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            harness = _Harness(tmp, "convnext")
            harness.model.unlink()
            rc, err, _ = harness.run()
            self.assertEqual(rc, 2)
            stored = harness.evidence()
            self.assertFalse(stored["passed"])
            self.assertEqual(stored["return_code"], 2)
            self.assertEqual(stored["error"]["type"], "ValueError")
            self.assertIn("Missing or empty model file", stored["error"]["message"])

    def test_existing_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            harness = _Harness(tmp, "convnext")
            rc, _, _ = harness.run()
            self.assertEqual(rc, 0)
            first = harness.evidence()
            rc, err, _ = harness.run()
            self.assertEqual(rc, 2)
            self.assertIn("must not already exist", err)
            # The first run's evidence is untouched by the rejected re-run.
            self.assertEqual(harness.evidence()["started_utc"], first["started_utc"])


class B3CompareFailureTests(unittest.TestCase):
    def _harness_with(self, tmp, sample, configure):
        harness = _Harness.__new__(_Harness)
        harness.__init__(tmp, sample)
        harness.configure = configure.__get__(harness)
        return harness

    def test_raw_shape_mismatch_between_sides_fails_comparison(self):
        def configure(self, index):
            if index == 0:
                return dict(logits=_base_logits().copy())
            return dict(logits=_base_logits().copy(), output_shape=(1, 1000))

        with tempfile.TemporaryDirectory() as td:
            harness = self._harness_with(Path(td), "convnext", configure)
            rc, _, _ = harness.run()
            self.assertEqual(rc, 1)
            stored = harness.evidence()
            self.assertFalse(stored["passed"])
            self.assertFalse(stored["comparison"]["checks"]["raw_shape_equal"])
            self.assertEqual(len(stored["arrays"]), 12)  # no raw diff file

    def test_legacy_raw_dtype_mismatch_fails_comparison(self):
        def configure(self, index):
            if index == 0:
                return dict(logits=_base_logits().copy(), output_dtype="F64")
            return dict(logits=_base_logits().copy())

        with tempfile.TemporaryDirectory() as td:
            harness = self._harness_with(Path(td), "edgenext", configure)
            rc, _, _ = harness.run()
            self.assertEqual(rc, 1)
            stored = harness.evidence()
            self.assertFalse(stored["comparison"]["checks"]["raw_dtype_equal"])
            self.assertTrue(stored["comparison"]["checks"]["raw_shape_equal"])

    def test_unified_binding_rejects_non_f32_output_as_execution_error(self):
        def configure(self, index):
            if index == 0:
                return dict(logits=_base_logits().copy())
            return dict(logits=_base_logits().copy(), output_dtype="F16")

        with tempfile.TemporaryDirectory() as td:
            harness = self._harness_with(Path(td), "fasternet", configure)
            rc, _, _ = harness.run()
            self.assertEqual(rc, 2)
            stored = harness.evidence()
            self.assertEqual(stored["return_code"], 2)
            self.assertEqual(stored["error"]["type"], "MetadataMismatchError")
            self.assertIsNotNone(stored["metadata"]["legacy"])

    def test_score_perturbation_beyond_tolerance_fails(self):
        def configure(self, index):
            logits = _base_logits().copy()
            if index == 1:
                logits[999] += np.float32(0.5)
            return dict(logits=logits)

        with tempfile.TemporaryDirectory() as td:
            harness = self._harness_with(Path(td), "fastvit", configure)
            rc, _, _ = harness.run()
            self.assertEqual(rc, 1)
            stored = harness.evidence()
            self.assertFalse(stored["passed"])
            checks = stored["comparison"]["checks"]
            self.assertTrue(checks["topk_ids_equal"])
            self.assertFalse(checks["topk_scores_within_tolerance"])
            self.assertGreater(
                stored["comparison"]["topk"]["max_common_abs_score_diff"], 1e-5)
            self.assertEqual(len(stored["arrays"]), 13)

    def test_topk_id_mismatch_fails_with_per_id_evidence(self):
        def configure(self, index):
            logits = _base_logits().copy()
            if index == 1:
                logits[[0, 999]] = logits[[999, 0]]
            return dict(logits=logits)

        with tempfile.TemporaryDirectory() as td:
            harness = self._harness_with(Path(td), "convnext", configure)
            rc, _, _ = harness.run()
            self.assertEqual(rc, 1)
            stored = harness.evidence()
            comparison = stored["comparison"]
            self.assertFalse(comparison["checks"]["topk_ids_equal"])
            rows = {row["id"]: row for row in comparison["topk"]["per_id"]}
            # Each side's exclusive Top-K ID is marked absent on the other side.
            self.assertEqual(rows[999]["legacy_source"], "topk")
            self.assertEqual(rows[999]["unified_source"], "absent")
            self.assertEqual(rows[0]["unified_source"], "topk")
            self.assertEqual(rows[0]["legacy_source"], "absent")
            self.assertIsNotNone(comparison["top8_per_id"]["legacy"])
            self.assertIsNotNone(comparison["top8_per_id"]["unified"])

    def test_exact_tie_is_recorded_and_never_passed(self):
        logits = _base_logits().copy()
        logits[995] = logits[994]  # exact tie straddling ranks 5 and 6 (k=5)

        def configure(self, index):
            return dict(logits=logits.copy())

        with tempfile.TemporaryDirectory() as td:
            harness = self._harness_with(Path(td), "edgenext", configure)
            rc, _, _ = harness.run()
            self.assertEqual(rc, 1)
            stored = harness.evidence()
            comparison = stored["comparison"]
            self.assertFalse(stored["passed"])
            self.assertTrue(comparison["tie"]["detected"])
            self.assertFalse(comparison["checks"]["no_ambiguous_exact_tie"])
            self.assertIn("never auto-relaxes a tie into a pass", comparison["tie"]["policy"])
            for side in ("legacy", "unified"):
                groups = comparison["tie"]["sides"][side]
                self.assertTrue(groups, side)
                tied = {value for group in groups for value in group["ids"]}
                self.assertEqual(tied, {994, 995})
                top8 = [row["id"] for row in comparison["top8_per_id"][side]]
                self.assertIn(994, top8)
                self.assertIn(995, top8)
                for group in groups:
                    for score in group["scores"]:
                        self.assertIsNotNone(score)

    def test_execution_failure_on_unified_side_retains_partial_evidence(self):
        boom = RuntimeError("simulated SDK failure")

        def configure(self, index):
            if index == 0:
                return dict(logits=_base_logits().copy())
            return dict(logits=_base_logits().copy(), run_error=boom)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            harness = self._harness_with(tmp, "fastvit", configure)
            rc, err, _ = harness.run()
            self.assertEqual(rc, 2)
            stored = harness.evidence()
            self.assertEqual(stored["error"], {"type": "RuntimeError",
                                               "message": "simulated SDK failure"})
            self.assertEqual(stored["return_code"], 2)
            self.assertFalse(stored["passed"])
            self.assertIsNotNone(stored["finished_utc"])
            evidence = tmp / "evidence"
            self.assertTrue((evidence / "legacy_input_data.npy").is_file())
            self.assertTrue((evidence / "legacy_raw_prob.npy").is_file())
            self.assertTrue((evidence / "legacy_topk_ids.npy").is_file())
            self.assertTrue((evidence / "unified_input_data.npy").is_file())
            self.assertFalse((evidence / "unified_topk_ids.npy").exists())

    def test_copy_hostile_board_quant_descriptor_survives_metadata_evidence(self):
        def configure(self, index):
            return dict(logits=_base_logits().copy(), quant=_BoardQuantParams())

        with tempfile.TemporaryDirectory() as td:
            harness = self._harness_with(Path(td), "convnext", configure)
            rc, _, _ = harness.run()
            self.assertEqual(rc, 0)
            stored = harness.evidence()
            for side in ("legacy", "unified"):
                quant = stored["metadata"][side]["output_quants"]["prob"]
                self.assertEqual(quant["quant_type"], "SCALE")
                self.assertEqual(quant["scale"], 0.25)
                self.assertEqual(quant["zero_point"], 7)
                self.assertEqual(quant["axis"], 3)

    def test_pin_verification_failure_refuses_to_run(self):
        with tempfile.TemporaryDirectory() as td:
            harness = _Harness(Path(td), "convnext")
            with patch.object(tool, "_pin_blob_sha256", return_value=("0" * 64, None)):
                rc, _, _ = harness.run()
            self.assertEqual(rc, 2)
            stored = harness.evidence()
            self.assertFalse(stored["source_closure"]["verified"])
            self.assertEqual(stored["error"]["type"], "ValueError")
            self.assertIn("does not match pin", stored["error"]["message"])
            self.assertTrue(stored["source_closure"]["files"])
            self.assertEqual(stored["return_code"], 2)

    def test_drifted_dependency_file_is_named_in_the_refusal(self):
        real_pin_sha = tool._pin_blob_sha256

        def drifted(pin_path):
            if pin_path == "utils/py_utils/file_io.py":
                return "deadbeef" * 8, None
            return real_pin_sha(pin_path)

        with tempfile.TemporaryDirectory() as td:
            harness = _Harness(Path(td), "edgenext")
            with patch.object(tool, "_pin_blob_sha256", side_effect=drifted):
                rc, _, _ = harness.run()
            self.assertEqual(rc, 2)
            stored = harness.evidence()
            self.assertFalse(stored["source_closure"]["verified"])
            self.assertIn("utils/py_utils/file_io.py", stored["error"]["message"])
            for entry in stored["source_closure"]["files"]:
                if entry["pin_path"] == "utils/py_utils/file_io.py":
                    self.assertEqual(entry["pin_sha256"], "deadbeef" * 8)
                    self.assertFalse(entry["matches_pin"])


class B3ComparePureRuleTests(unittest.TestCase):
    @staticmethod
    def _record(ids, scores, evidence_ids=None, evidence_scores=None):
        return {
            "inputs": {"data": np.zeros(8, dtype=np.uint8)},
            "outputs": {"prob": np.zeros(5, dtype=np.float32)},
            "topk": {
                "ids": np.asarray(ids),
                "scores": np.asarray(scores),
                "labels": [f"class-{int(value)}" for value in ids],
            },
            "evidence": {
                "ids": np.asarray(evidence_ids if evidence_ids is not None else ids),
                "scores": np.asarray(evidence_scores if evidence_scores is not None else scores),
            },
        }

    def test_input_bytes_equal_across_different_shapes(self):
        flat = np.arange(12, dtype=np.uint8)
        record = {
            "outputs": {},
            "topk": {"ids": np.asarray([0]), "scores": np.asarray([1.0]),
                     "labels": ["class-0"]},
            "evidence": {"ids": np.asarray([0]), "scores": np.asarray([1.0])},
        }
        legacy = dict(record, inputs={"data": flat.reshape(1, 12, 1)})
        unified = dict(record, inputs={"data": flat.copy()})
        result = tool.compare_records(legacy, unified, top_k=1)
        self.assertTrue(result["checks"]["inputs_exact_bytes"])
        self.assertEqual(result["inputs"]["data"]["legacy_shape"], [1, 12, 1])
        self.assertEqual(result["inputs"]["data"]["unified_shape"], [12])

    def test_input_byte_difference_fails(self):
        legacy = self._record([0], [1.0])
        legacy["inputs"]["data"][3] = 9
        unified = self._record([0], [1.0])
        result = tool.compare_records(legacy, unified, top_k=1)
        self.assertFalse(result["checks"]["inputs_exact_bytes"])

    def test_score_tolerance_is_inclusive_and_nonfinite_fails(self):
        legacy = self._record([0, 1], [0.5, 0.25])
        unified = self._record([0, 1], [0.5 + 1e-5, 0.25])
        self.assertTrue(tool.compare_records(legacy, unified, top_k=2)["passed"])
        unified = self._record([0, 1], [0.5 + 1.1e-5, 0.25])
        self.assertFalse(
            tool.compare_records(legacy, unified, top_k=2)["checks"]["topk_scores_within_tolerance"])
        unified = self._record([0, 1], [np.nan, 0.25])
        self.assertFalse(
            tool.compare_records(legacy, unified, top_k=2)["checks"]["topk_scores_within_tolerance"])

    def test_empty_inputs_or_outputs_cannot_pass_vacuously(self):
        legacy = self._record([0], [1.0])
        unified = self._record([0], [1.0])
        legacy["inputs"] = {}
        result = tool.compare_records(legacy, unified, top_k=1)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["input_names_equal"])
        self.assertFalse(result["checks"]["inputs_exact_bytes"])
        self.assertFalse(result["checks"]["inputs_uint8_finite"])
        legacy = self._record([0], [1.0])
        unified = self._record([0], [1.0])
        legacy["outputs"] = {}
        result = tool.compare_records(legacy, unified, top_k=1)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["raw_output_names_equal"])
        self.assertFalse(result["checks"]["raw_shape_equal"])
        self.assertFalse(result["checks"]["raw_dtype_equal"])

    def test_non_uint8_inputs_fail_the_dtype_gate(self):
        legacy = self._record([0], [1.0])
        unified = self._record([0], [1.0])
        for record in (legacy, unified):
            record["inputs"]["data"] = record["inputs"]["data"].astype(np.float32)
        result = tool.compare_records(legacy, unified, top_k=1)
        self.assertFalse(result["checks"]["inputs_uint8_finite"])
        self.assertFalse(result["passed"])
        # Byte equivalence across view shapes remains the input criterion.
        self.assertTrue(result["checks"]["inputs_exact_bytes"])

    def test_topk_count_and_id_uniqueness_are_enforced(self):
        short = self._record([0, 1, 2, 3], [0.4, 0.3, 0.2, 0.1])
        result = tool.compare_records(short, short, top_k=5)
        self.assertFalse(result["checks"]["topk_counts_match_request"])
        self.assertFalse(result["passed"])
        duplicated = self._record([2, 2], [0.5, 0.5])
        result = tool.compare_records(duplicated, duplicated, top_k=2)
        self.assertFalse(result["checks"]["topk_ids_unique"])
        self.assertFalse(result["passed"])

    def test_label_mismatch_fails_even_with_identical_ids_and_scores(self):
        legacy = self._record([0, 1], [0.6, 0.4])
        unified = self._record([0, 1], [0.6, 0.4])
        unified["topk"]["labels"] = ["renamed-0", "class-1"]
        result = tool.compare_records(legacy, unified, top_k=2)
        self.assertTrue(result["checks"]["topk_ids_equal"])
        self.assertTrue(result["checks"]["topk_scores_within_tolerance"])
        self.assertFalse(result["checks"]["topk_labels_equal"])
        self.assertFalse(result["passed"])

    def test_tie_window_covers_boundary_only(self):
        # Tie between ranks 4 and 5 with k=5 (membership ambiguity) is caught.
        legacy = self._record(
            [7, 6, 5, 4, 3, 2], [0.9, 0.8, 0.7, 0.6, 0.5, 0.5],
            evidence_ids=[7, 6, 5, 4, 3, 2, 1, 0],
            evidence_scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.3],
        )
        result = tool.compare_records(legacy, legacy, top_k=5)
        self.assertTrue(result["tie"]["detected"])
        self.assertEqual(result["tie"]["sides"]["legacy"][0]["ids"], [3, 2])
        # A tie strictly below the boundary rank cannot change the Top-K.
        deep = self._record(
            [7, 6, 5, 4, 3], [0.9, 0.8, 0.7, 0.6, 0.5],
            evidence_ids=[7, 6, 5, 4, 3, 2, 1, 0],
            evidence_scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.2, 0.2, 0.1],
        )
        result = tool.compare_records(deep, deep, top_k=5)
        self.assertFalse(result["tie"]["detected"])
        self.assertTrue(result["checks"]["no_ambiguous_exact_tie"])

    def test_raw_difference_is_reported(self):
        legacy = self._record([0], [1.0])
        legacy["outputs"]["prob"] = np.array([1.0, 2.0], dtype=np.float32)
        unified = self._record([0], [1.0])
        unified["outputs"]["prob"] = np.array([1.0, 2.5], dtype=np.float32)
        result = tool.compare_records(legacy, unified, top_k=1)
        self.assertTrue(result["checks"]["raw_shape_equal"])
        raw = result["raw_outputs"]["prob"]
        self.assertEqual(raw["nonzero_diff_count"], 1)
        self.assertEqual(raw["max_abs_diff"], 0.5)
        self.assertEqual(float(np.max(result["raw_diffs"]["prob"])), 0.5)


class B3CompareParserTests(unittest.TestCase):
    def test_tool_parses_under_board_python_3_10(self):
        """The board images run Python 3.10; guard against newer-only syntax."""

        import ast

        source = (TESTS_DIR.parent / "b3_classification_compare.py").read_text(
            encoding="utf-8"
        )
        ast.parse(source, feature_version=(3, 10))

    def test_defaults_follow_sample_semantics(self):
        parser = tool.build_parser()
        args = parser.parse_args([
            "--sample", "convnext", "--target", "x5", "--output-dir", "/tmp/new"
        ])
        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.resize_type, 1)
        self.assertEqual(args.priority, 0)
        self.assertEqual(args.bpu_cores, [0])
        self.assertIsNone(args.variant)
        self.assertIsNone(args.asset_id)
        self.assertIsNone(args.model_path)
        self.assertIsNone(args.test_img)
        self.assertIsNone(args.label_file)

    def test_default_image_per_sample_exists_in_the_checkout(self):
        for sample, spec in tool.SAMPLES.items():
            with self.subTest(sample=sample):
                image = (
                    tool._ROOT / "samples" / "vision" / sample
                    / "test_data" / spec["test_image"]
                )
                self.assertTrue(image.is_file(), image)
        self.assertTrue(
            (tool._ROOT / "datasets" / "imagenet" / "imagenet_classes.names").is_file())


if __name__ == "__main__":
    unittest.main()
