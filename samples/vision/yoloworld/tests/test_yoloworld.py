# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Host-only source protocol and prompt/context tests."""
import json, types, unittest, importlib.util, sys
from pathlib import Path
import numpy as np
from samples.vision.yoloworld.runtime.python.model_binding import resolve_selection, bind_model
from samples.vision.yoloworld.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.yoloworld.runtime.python.yoloworld import YOLOWorldTask
from samples._shared.runtime_meta import RuntimeMetadata
from unittest.mock import patch
SAMPLE=Path(__file__).resolve().parents[1]

class FakeRuntime:
    model_names=['yolo']
    input_names={'yolo':['image','text']}; input_shapes={'yolo':{'image':(1,3,640,640),'text':(1,32,512,1)}}
    input_dtypes={'yolo':{'image':'float32','text':'float32'}}
    output_names={'yolo':['scores','boxes']}; output_shapes={'yolo':{'scores':(1,8400,32),'boxes':(1,8400,4)}}
    output_dtypes={'yolo':{'scores':'float32','boxes':'float32'}}
    def __init__(self): self.calls=[]; self.scores=np.zeros((1,8400,32),np.float32); self.boxes=np.zeros((1,8400,4),np.float32); self.scheduling=None
    def run(self,inputs): self.calls.append(inputs); return {'yolo':{'scores':self.scores,'boxes':self.boxes}}
    def set_scheduling_params(self,**kwargs): self.scheduling=kwargs

class BoardQuantParams:
    """Mimics hbm_runtime.QuantParams: attributes read, any copy refuses (X5 board evidence 2026-09-24)."""
    def __init__(self,quant_type='NONE',scale=1.0,zero_point=0,axis=0):
        self.quant_type=types.SimpleNamespace(name=quant_type);self.scale=np.asarray(scale,np.float32);self.zero_point=np.asarray(zero_point,np.int32);self.axis=axis
    def __deepcopy__(self,memo):raise TypeError("cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object")
    def __copy__(self):raise TypeError("cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object")

def fixture():
    vocab=json.loads((SAMPLE/'test_data/offline_vocabulary_embeddings.json').read_text())
    runtime=FakeRuntime(); sel=resolve_selection('x5'); runner=RuntimeModelRunner(sel,runtime=runtime); binding=runner.load()
    return YOLOWorldTask(runner,binding,vocab),runtime

class YOLOWorldTests(unittest.TestCase):
    def test_asset_identity_and_metadata_binding(self):
        sel=resolve_selection('x5'); self.assertEqual(sel.asset.reference,'x5:yoloworld:yolo_world.bin')
        with self.assertRaises(ValueError): resolve_selection('x5',model_path='/tmp/custom.bin')
        meta=RuntimeMetadata.from_runtime(FakeRuntime()); self.assertEqual(bind_model(sel,meta).score_output_name,'scores')
        bad=FakeRuntime();bad.output_shapes={'yolo':{'scores':(1,8400,80),'boxes':(1,8400,4)}}
        with self.assertRaises(ValueError): bind_model(sel,RuntimeMetadata.from_runtime(bad))

    def test_source_geometry_dtype_slots_and_empty_prompt(self):
        task,_=fixture(); image=np.zeros((720,1280,3),np.uint8)
        prepared=task.pre_process(image,['dog']); self.assertEqual(prepared.tensors[task.binding.image_input_name].shape,(1,3,640,640))
        self.assertEqual(prepared.tensors[task.binding.image_input_name].dtype,np.float32)
        self.assertEqual(prepared.tensors[task.binding.text_input_name].shape,(1,32,512,1)); self.assertEqual(prepared.context.class_ids[-1],task.class_names.index('dog'))
        with self.assertRaises(ValueError): task.pre_process(image,[])
        with self.assertRaises(ValueError): task.pre_process(image,[''])
        with self.assertRaises(KeyError): task.pre_process(image,['not-in-vocabulary'])
        with self.assertRaises(ValueError): task.pre_process(image,['dog']*33)

    def test_predict_equals_explicit_stages_and_ab_a_context(self):
        task,runtime=fixture(); rng=np.random.default_rng(4)
        runtime.scores[0,7,0]=.9;runtime.boxes[0,7]=[1,2,100,200]
        a=rng.integers(0,256,(20,50,3),np.uint8); b=rng.integers(0,256,(70,30,3),np.uint8)
        pa=task.pre_process(a,['dog']); saved=pa.tensors[task.binding.image_input_name].copy()
        pb=task.pre_process(b,['person','dog']); again=task.pre_process(a,['dog'])
        self.assertNotEqual(pa.context,pb.context); self.assertEqual(pa.context,again.context); np.testing.assert_array_equal(pa.tensors[task.binding.image_input_name],saved)
        explicit=task.post_process(task.forward(pa),pa.context); composed=task.predict(a,['dog'])
        np.testing.assert_array_equal(explicit.boxes,composed.boxes); np.testing.assert_array_equal(explicit.class_ids,composed.class_ids)
        self.assertEqual(runtime.calls[-1]['yolo']['image'].shape,(1,3,640,640))

    def test_preprocess_and_postprocess_match_fixed_source_fixture(self):
        task, runtime = fixture()
        spec = importlib.util.spec_from_file_location("yoloworld_fixed_source", SAMPLE.parent.parent.parent / "platforms/x5/samples/vision/yoloworld/runtime/python/yoloworld_det.py")
        legacy = importlib.util.module_from_spec(spec)
        fake_hbm = types.ModuleType("hbm_runtime")
        fake_hbm.QuantParams = type("QuantParams", (), {})
        with patch.dict(sys.modules, {"hbm_runtime": fake_hbm, spec.name: legacy}):
            assert spec.loader is not None
            spec.loader.exec_module(legacy)
        old = legacy.YOLOWorldDetect.__new__(legacy.YOLOWorldDetect)
        old.cfg = legacy.YOLOWorldConfig("unused", str(SAMPLE / "test_data/offline_vocabulary_embeddings.json"))
        old.model_name = "yolo"; old.input_names = ["image", "text"]; old.output_names = ["scores", "boxes"]
        old.vocabulary = json.loads((SAMPLE / "test_data/offline_vocabulary_embeddings.json").read_text())
        old.class_names = list(old.vocabulary)
        image = np.random.default_rng(11).integers(0, 256, (37, 91, 3), dtype=np.uint8)
        source_inputs = old.pre_process(image, ["person", "dog"]) ["yolo"]
        prepared = task.pre_process(image, ["person", "dog"])
        np.testing.assert_array_equal(prepared.tensors["image"], source_inputs["image"])
        np.testing.assert_array_equal(prepared.tensors["text"], source_inputs["text"])
        runtime.scores.fill(0); runtime.boxes.fill(0)
        runtime.scores[0, 13, 0] = .8; runtime.scores[0, 21, 1] = .7
        runtime.boxes[0, 13] = [1, 2, 30, 20]; runtime.boxes[0, 21] = [2, 3, 31, 21]
        old._scale = prepared.context.scale; old._selected_class_ids = np.asarray(prepared.context.class_ids, dtype=np.int32)
        source_result = old.post_process({"yolo": {"scores": runtime.scores[..., None], "boxes": runtime.boxes[..., None]}}, image.shape[1], image.shape[0])
        unified_result = task.post_process({"scores": runtime.scores, "boxes": runtime.boxes}, prepared.context)
        for got, expected in zip(unified_result.boxes, source_result[0]): np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(unified_result.scores, source_result[1]); np.testing.assert_array_equal(unified_result.class_ids, source_result[2])

    def test_runner_preserves_native_outputs_and_rejects_wrong_input(self):
        task,runtime=fixture(); prepared=task.pre_process(np.zeros((10,10,3),np.uint8),['dog']); raw=task.forward(prepared)
        self.assertIs(raw[task.binding.score_output_name],runtime.scores); self.assertIs(raw[task.binding.box_output_name],runtime.boxes)
        with self.assertRaises(ValueError): task.forward({'wrong':prepared.tensors[task.binding.image_input_name]})

    def test_vocabulary_is_a_read_only_owned_snapshot(self):
        """A caller mutating its own embedding array must not change later calls."""
        vocab = json.loads((SAMPLE / 'test_data/offline_vocabulary_embeddings.json').read_text())
        key = next(iter(vocab))
        caller_array = np.asarray(vocab[key], dtype=np.float32)
        supplied = dict(vocab)
        supplied[key] = caller_array
        runtime = FakeRuntime()
        runner = RuntimeModelRunner(resolve_selection('x5'), runtime=runtime)
        task = YOLOWorldTask(runner, runner.load(), supplied)
        image = np.zeros((40, 60, 3), np.uint8)
        before = task.pre_process(image, [key]).tensors[task.binding.text_input_name].copy()
        self.assertFalse(task.vocabulary[key].flags.writeable)
        self.assertIsNot(task.vocabulary[key], caller_array)
        caller_array[:] = 12345.0
        np.testing.assert_array_equal(task.pre_process(image, [key]).tensors[task.binding.text_input_name], before)

    def test_binding_rejects_a_forged_publication_row(self):
        import dataclasses
        sel = resolve_selection('x5')
        forged_asset = dataclasses.replace(sel.asset, sha256='0' * 64, url='https://example.invalid/evil.bin')
        forged = type(sel)(sel.target, forged_asset, Path('/tmp/evil.bin'))
        with self.assertRaises(ValueError):
            bind_model(forged, RuntimeMetadata.from_runtime(FakeRuntime()))

    def test_runner_rejects_non_finite_outputs(self):
        from samples._shared.runtime_meta import MetadataMismatchError
        runtime = FakeRuntime()
        runtime.scores[0, 0, 0] = np.nan
        runner = RuntimeModelRunner(resolve_selection('x5'), runtime=runtime)
        binding = runner.load()
        tensors = {
            binding.image_input_name: np.zeros((1, 3, 640, 640), np.float32),
            binding.text_input_name: np.zeros((1, 32, 512, 1), np.float32),
        }
        with self.assertRaises(MetadataMismatchError):
            runner(tensors)

    def test_real_path_gates_before_sdk_and_the_seam_skips_the_gate(self):
        # The gate import is function-local, so patch the shared function itself.
        import samples.vision.yoloworld.runtime.python.model_runner as runner_mod
        sel = resolve_selection('x5')
        with patch('samples._shared.platforms.require_execution_target',
                   side_effect=ValueError('no board identity')) as shared_gate:
            with self.assertRaises(ValueError):
                runner_mod.RuntimeModelRunner(sel).load()
            shared_gate.assert_called_once_with('x5')
        with patch('samples._shared.platforms.require_execution_target',
                   side_effect=AssertionError('injected factory is the host seam')):
            runner = runner_mod.RuntimeModelRunner(sel, runtime_factory=lambda path: FakeRuntime())
            self.assertIsNotNone(runner.load())

class YOLOWorldEvaluatorTests(unittest.TestCase):
    """The evaluator must run both sides itself, not compare hand-made files."""

    def _fixtures(self, temp):
        model = Path(temp) / "yolo_world.bin"
        model.write_bytes(b"fixture-yoloworld-model")
        selection = resolve_selection(
            "x5", model_path=str(model), asset_id="x5:yoloworld:yolo_world.bin"
        )
        image_path = SAMPLE / "test_data/dog.jpeg"
        image = __import__("cv2").imread(str(image_path), __import__("cv2").IMREAD_COLOR)
        self.assertIsNotNone(image)
        vocab_path = SAMPLE / "test_data/offline_vocabulary_embeddings.json"
        vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
        return selection, image, image_path, vocabulary, vocab_path

    @staticmethod
    def _runtime(score_slot: int = 0) -> "FakeRuntime":
        # The fixed source squeezes the terminal singleton, so the recorded
        # native tensors keep the exported (1,8400,32,1)/(1,8400,4,1) shape.
        runtime = FakeRuntime()
        runtime.scores = np.zeros((1, 8400, 32, 1), np.float32)
        runtime.boxes = np.zeros((1, 8400, 4, 1), np.float32)
        runtime.scores[0, 13, score_slot, 0] = 0.8
        runtime.scores[0, 21, score_slot, 0] = 0.7
        runtime.boxes[0, 13, :, 0] = [1, 2, 30, 20]
        runtime.boxes[0, 21, :, 0] = [2, 3, 31, 21]
        return runtime

    def _factory(self, runtimes):
        state = {"index": 0}

        def make(path):
            value = runtimes[min(state["index"], len(runtimes) - 1)]
            state["index"] += 1
            return value

        return make

    def _run(self, compare, selection, image, image_path, vocabulary, vocab_path,
             directory, factory):
        sdk = types.ModuleType("hbm_runtime")
        sdk.HB_HBMRuntime = FakeRuntime
        old_sdk = sys.modules.get("hbm_runtime")
        sys.modules["hbm_runtime"] = sdk
        try:
            with patch.object(compare, "require_execution_target", return_value="x5"):
                return compare.run_comparison(
                    selection, image, image_path, ["dog"], vocabulary, vocab_path, directory,
                    runtime_factory=factory,
                )
        finally:
            if old_sdk is None:
                sys.modules.pop("hbm_runtime", None)
            else:
                sys.modules["hbm_runtime"] = old_sdk

    def test_evaluator_captures_both_sides_with_complete_identity(self):
        import importlib
        compare = importlib.import_module("samples.vision.yoloworld.evaluator.compare")
        with __import__("tempfile").TemporaryDirectory() as temp:
            selection, image, image_path, vocabulary, vocab_path = self._fixtures(temp)
            directory = Path(temp) / "success"
            summary = self._run(compare, selection, image, image_path, vocabulary, vocab_path,
                                directory, self._factory([self._runtime()]))
            self.assertEqual(summary["return_code"], 0, summary.get("error"))
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["prompts"], ["dog"])
            self.assertTrue(summary["model_sha256"] and summary["image_sha256"])
            self.assertTrue(summary["vocabulary_sha256"] and summary["code_sha256"])
            self.assertEqual(set(summary["metadata"]), {"legacy", "unified"})
            self.assertTrue(summary["started_utc"] and summary["finished_utc"])
            self.assertTrue(summary["argv"] and summary["cwd"])
            for filename, entry in summary["arrays"].items():
                self.assertTrue((directory / filename).is_file(), filename)
                self.assertEqual(len(entry["sha256"]), 64)
            self.assertTrue((directory / "comparison.json").is_file())
            self.assertTrue(all(summary["checks"].values()))

    def test_evaluator_metadata_survives_copy_hostile_board_quant_params(self):
        """The old asdict() metadata snapshot raised TypeError on the real board."""
        import importlib
        compare = importlib.import_module("samples.vision.yoloworld.evaluator.compare")
        def quant_runtime(_):
            runtime = self._runtime()
            runtime.output_quants = {'yolo': {'scores': BoardQuantParams(), 'boxes': BoardQuantParams()}}
            return runtime
        with __import__("tempfile").TemporaryDirectory() as temp:
            selection, image, image_path, vocabulary, vocab_path = self._fixtures(temp)
            directory = Path(temp) / "quants"
            summary = self._run(compare, selection, image, image_path, vocabulary, vocab_path,
                                directory, quant_runtime)
            self.assertEqual(summary["return_code"], 0, summary.get("error"))
            self.assertTrue(summary["passed"])
            for side in ("legacy", "unified"):
                quants = summary["metadata"][side]["output_quants"]
                for name in ("scores", "boxes"):
                    self.assertEqual(quants[name]["quant_type"], "NONE")
                    self.assertEqual(quants[name]["scale"], 1.0)
                    self.assertEqual(quants[name]["zero_point"], 0)
                    self.assertEqual(quants[name]["axis"], 0)
            saved = json.loads((directory / "comparison.json").read_text())
            self.assertEqual(saved["metadata"]["unified"]["output_quants"]["scores"]["scale"], 1.0)

    def test_evaluator_reports_a_real_difference_instead_of_passing(self):
        import importlib
        compare = importlib.import_module("samples.vision.yoloworld.evaluator.compare")
        changed = self._runtime()
        changed.boxes[0, 13, :, 0] = [5, 6, 300, 200]
        with __import__("tempfile").TemporaryDirectory() as temp:
            selection, image, image_path, vocabulary, vocab_path = self._fixtures(temp)
            directory = Path(temp) / "difference"
            summary = self._run(compare, selection, image, image_path, vocabulary, vocab_path,
                                directory, self._factory([self._runtime(), changed]))
            self.assertEqual(summary["return_code"], 1)
            self.assertFalse(summary["passed"])
            self.assertFalse(summary["checks"]["result.boxes"])
            self.assertTrue((directory / "comparison.json").is_file())

    def test_evaluator_records_execution_failure_and_still_writes_evidence(self):
        import importlib
        compare = importlib.import_module("samples.vision.yoloworld.evaluator.compare")
        with __import__("tempfile").TemporaryDirectory() as temp:
            selection, image, image_path, vocabulary, vocab_path = self._fixtures(temp)
            directory = Path(temp) / "error"

            def explode(path):
                raise RuntimeError("fake SDK failure")

            with self.assertRaises(RuntimeError):
                self._run(compare, selection, image, image_path, vocabulary, vocab_path,
                          directory, explode)
            payload = json.loads((directory / "comparison.json").read_text())
            self.assertEqual(payload["return_code"], 2)
            self.assertEqual(payload["error"]["type"], "RuntimeError")
            self.assertFalse(payload["passed"])

    def test_prompt_parsing_rejects_empty_and_overflow(self):
        import importlib
        compare = importlib.import_module("samples.vision.yoloworld.evaluator.compare")
        for text in ("", "dog,,cat", ","):
            with self.assertRaises(ValueError):
                compare._parse_prompts(text)
        with self.assertRaises(ValueError):
            compare._parse_prompts(",".join(["dog"] * 33))
        self.assertEqual(compare._parse_prompts("dog, person"), ["dog", "person"])


if __name__=='__main__': unittest.main()
