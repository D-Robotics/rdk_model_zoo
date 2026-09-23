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

if __name__=='__main__': unittest.main()
