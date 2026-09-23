"""Offline contracts and fixed-source numerical regression for YOLOv5."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from samples._shared.runtime_meta import RuntimeMetadata, MetadataMismatchError
from samples.vision.yolov5.runtime.python.model_binding import resolve_selection, bind_model, list_available_assets
from samples.vision.yolov5.runtime.python.detection import YOLOv5Task

ROOT = Path(__file__).resolve().parents[4]


def facts(target='x5', dtype='float32'):
    size = 640 if target == 'x5' else 672
    ins = {'images': (1, 3, size, size)} if target == 'x5' else {'images_y': (1, size, size, 1), 'images_uv': (1, size//2, size//2, 2)}
    names = ('small', 'medium', 'large')
    return dict(model_name='detector', input_names=tuple(ins), input_shapes=ins,
                input_dtypes={n: 'uint8' for n in ins}, output_names=names,
                output_shapes={n: (1, size//s, size//s, 255) for n, s in zip(names, (8,16,32))},
                output_dtypes={n: dtype for n in names},
                output_quants={n: types.SimpleNamespace(quant_type='SCALE' if dtype!='float32' else 'NONE', scale=np.array([.1],np.float32), zero_point=np.array([0]), axis=3) for n in names})


class FakeRuntime:
    def __init__(self, target='x5', dtype='float32'):
        self.facts = facts(target, dtype)
        self.model_names = ['detector']
        for k, v in self.facts.items():
            if k != 'model_name': setattr(self, k, {'detector': v})
        self.outputs = {n: np.full(shape, -80 if dtype != 'float32' else -8., dtype=dtype) for n,shape in self.facts['output_shapes'].items()}
        # Three detections with distinct class / score / geometry across heads.
        for i, n in enumerate(self.facts['output_names']):
            a=self.outputs[n].reshape(-1,85); a[3+i,:4]=0; a[3+i,4]=6 if dtype=='float32' else 60; a[3+i,5+i]=5 if dtype=='float32' else 50
        self.calls=[]
    def run(self, x):
        self.calls.append(x); return {'detector': self.outputs}
    def set_scheduling_params(self, **kw): self.schedule=kw


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


def legacy(target, runtime, resize=None):
    group='x5' if target=='x5' else 's'
    old=sys.modules.get('hbm_runtime')
    sys.modules['hbm_runtime']=types.SimpleNamespace(HB_HBMRuntime=lambda _:runtime,QuantParams=object)
    try:
        p=ROOT/f'platforms/{group}/samples/vision/yolov5/runtime/python'/('yolov5_det.py' if group=='x5' else 'yolov5.py')
        mod=load(p,'legacy_y5_'+group)
        mod.pre_utils=load(ROOT/f'platforms/{group}/utils/py_utils/preprocess.py','legacy_y5_pre_'+group)
        mod.post_utils=load(ROOT/f'platforms/{group}/utils/py_utils/postprocess.py',f'platforms.{group}.utils.py_utils.postprocess')
    finally:
        if old is None:sys.modules.pop('hbm_runtime',None)
        else:sys.modules['hbm_runtime']=old
    cfg=mod.YOLOv5Config('fixture',**({'resize_type':resize} if resize is not None else {}))
    return (mod.YOLOv5Detect if group=='x5' else mod.YoloV5X)(cfg)


class YOLOv5Tests(unittest.TestCase):
    def test_per_channel_dequant_is_source_exact_and_descriptor_is_immutable(self):
        runtime=FakeRuntime('s100','int8')
        for name in runtime.facts['output_names']:
            runtime.facts['output_quants'][name].scale=np.linspace(.08,.12,255,dtype=np.float32)
        task=YOLOv5Task(lambda _:runtime.outputs,bind_model(resolve_selection('s100'),RuntimeMetadata.from_mapping(runtime.facts)))
        old=legacy('s100',runtime)
        image=np.zeros((149,97,3),np.uint8)
        expected=old.predict(image)
        result=task.predict(image)
        for actual,want in zip((result.boxes,result.scores,result.class_ids),expected):np.testing.assert_array_equal(actual,want)
        quant=task.binding.output_quants['small']
        with self.assertRaises(ValueError):quant.scale[0]=9
        runtime.facts['output_quants']['small'].scale.fill(9)
        np.testing.assert_array_equal(task.predict(image).boxes,result.boxes)

    def test_invalid_context_and_multimodel_metadata_rejected(self):
        from samples.vision.yolov5.runtime.python.tensor_io import DetectionContext
        for args in [((0,1),640,0),([10,20],640,0),((1,2),639,0),((1,2),640,2)]:
            with self.assertRaises(ValueError):DetectionContext(*args)
        meta=facts('x5');meta['model_names']=('detector','other')
        with self.assertRaises(MetadataMismatchError):bind_model(resolve_selection('x5'),meta)

    def make_task(self,target='x5',dtype='float32'):
        runtime=FakeRuntime(target,dtype); b=bind_model(resolve_selection(target),RuntimeMetadata.from_mapping(runtime.facts))
        return YOLOv5Task(lambda tensors:runtime.outputs,b),runtime,b

    def test_all_source_assets_and_defaults(self):
        self.assertEqual(len(list_available_assets('x5')),9)
        self.assertIn('yolov5n_tag_v7.0',resolve_selection('x5').asset.filename)
        for t in ('s100','s600'): self.assertEqual(resolve_selection(t).asset.filename,f'{t}/yolov5x_672x672_nv12.hbm')
        with self.assertRaises(ValueError):resolve_selection('s100p')
        self.assertEqual(resolve_selection('s100p',consumer='bytetrack').asset.sample_id,'bytetrack')

    def test_custom_path_identity_and_mismatch(self):
        with self.assertRaises(ValueError):resolve_selection('x5',model_path='/tmp/model')
        a=resolve_selection('x5').asset
        self.assertEqual(resolve_selection('x5',model_path='/tmp/model',asset_id=a.reference).model_path,Path('/tmp/model'))
        with self.assertRaises(ValueError):resolve_selection('s100',asset_id=a.reference)

    def test_binding_checks_shapes_native_dtype_and_quantization(self):
        for target in ('x5','s100','s600'):
            s=resolve_selection(target); m=facts(target);b=bind_model(s,RuntimeMetadata.from_mapping(m))
            self.assertEqual(b.output_transform,'raw_f32' if target=='x5' else 'dequant')
            m['output_shapes']['small']=(1,20,20,255)
            with self.assertRaises(MetadataMismatchError):bind_model(s,RuntimeMetadata.from_mapping(m))
        with self.assertRaises(MetadataMismatchError):bind_model(resolve_selection('x5'),RuntimeMetadata.from_mapping(facts('x5','int8')))
        m=facts('s100','int8');m['output_quants']={}
        with self.assertRaises(MetadataMismatchError):bind_model(resolve_selection('s100'),RuntimeMetadata.from_mapping(m))

    def test_fixed_source_inputs_and_results(self):
        image=np.arange(97*151*3,dtype=np.uint8).reshape(97,151,3)
        for target,dtype in [('x5','float32'),('s100','float32'),('s600','int8')]:
            for resize in (0,1):
                with self.subTest(target=target,dtype=dtype,resize=resize):
                    task,runtime,b=self.make_task(target,dtype);old=legacy(target,runtime,resize)
                    p=task.pre_process(image,resize_type=resize);old_inputs=old.pre_process(image)['detector']
                    for n,a in p.tensors.items():np.testing.assert_array_equal(a,old_inputs[n])
                    actual=task.post_process(task.forward(p.tensors),p.context)
                    expected=old.post_process({'detector':runtime.outputs},151,97)
                    if target=='x5':
                        boxes=np.array([x[2:] for x in expected],np.float32).reshape(-1,4);scores=np.array([x[1] for x in expected],np.float32);ids=np.array([x[0] for x in expected],np.int32)
                    else:boxes,scores,ids=expected
                    np.testing.assert_array_equal(actual.boxes,boxes);np.testing.assert_array_equal(actual.scores,scores);np.testing.assert_array_equal(actual.class_ids,ids)

    def test_predict_explicit_stages_and_context_aba(self):
        for target in ('x5','s100'):
            task,runtime,b=self.make_task(target)
            a=np.zeros((97,151,3),np.uint8);z=np.zeros((151,97,3),np.uint8)
            pa=task.pre_process(a,resize_type=0);pb=task.pre_process(z,resize_type=1)
            ra=task.post_process(task.forward(pa.tensors),pa.context)
            task.post_process(task.forward(pb.tensors),pb.context)
            again=task.predict(a,resize_type=0)
            np.testing.assert_array_equal(ra.boxes,again.boxes)
            self.assertEqual(pa.context.original_size,(97,151));self.assertEqual(pa.context.resize_type,0)
            self.assertEqual(pb.context.resize_type,1)
            with self.assertRaises(Exception):pa.context.resize_type=1
            raw=task.forward(pa.tensors)
            for name in raw:self.assertIs(raw[name],runtime.outputs[name])

    def test_native_mismatch_is_rejected_and_results_owned(self):
        task,runtime,b=self.make_task('s100','int8');p=task.pre_process(np.zeros((45,71,3),np.uint8))
        result=task.post_process(task.forward(p.tensors),p.context);boxes=result.boxes.copy()
        for x in runtime.outputs.values():x.fill(0)
        np.testing.assert_array_equal(result.boxes,boxes)
        runtime.outputs['small']=runtime.outputs['small'].astype(np.float32)
        with self.assertRaises(MetadataMismatchError):task.forward(p.tensors)

    def test_zero_threshold_override_is_not_replaced_with_default(self):
        task,runtime,b=self.make_task('s100');p=task.pre_process(np.zeros((45,71,3),np.uint8))
        # Keep only three non-underflow detections; source `or` bug swallowed zero NMS.
        raw=task.forward(p.tensors)
        with patch('samples.vision.yolov5.runtime.python.decode.classwise_nms',return_value=np.array([],dtype=int)) as nms:
            task.post_process(raw,p.context,score_thres=0,nms_thres=0)
            self.assertEqual(nms.call_args.args[-1],0)

if __name__=='__main__':unittest.main()
