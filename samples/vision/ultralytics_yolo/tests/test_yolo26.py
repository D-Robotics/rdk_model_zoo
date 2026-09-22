"""YOLO26 family contracts through the unified sample, without board runtime."""
from pathlib import Path
import sys, unittest, json
import numpy as np
import importlib, importlib.util, types
import subprocess, tempfile
from unittest.mock import patch, MagicMock
S = Path(__file__).resolve().parents[1]
R = S.parents[2]
sys.path.insert(0, str(S/'runtime/python'))

class Yolo26Contracts(unittest.TestCase):
    def test_pose_visibility_preserves_platform_convention(self):
        sys.path.insert(0,str(S/'evaluator'))
        from eval_yolo_pose import flatten_keypoints
        xy=np.zeros((3,2)); confidence=np.array([0,.5,.9])
        self.assertEqual(flatten_keypoints(xy,confidence)[2::3],[0,1,1])
        self.assertEqual(flatten_keypoints(xy,confidence,yolo26_platform='x5')[2::3],[1,2,2])
        self.assertEqual(flatten_keypoints(xy,confidence,yolo26_platform='s')[2::3],[1,1,1])

    def test_all_task_platform_input_bindings(self):
        from yolo_dispatch import get_task_types
        from yolo_platform import resolve_platform
        for platform in ('x5','s100','s100p','s600'):
            profile=resolve_platform(platform)
            for task in ('detect','cls','seg','pose','obb'):
                with self.subTest(platform=platform,task=task):
                    height=224 if task=='cls' else 64
                    names=['image'] if platform=='x5' else ['y','uv']
                    shapes={'image':(1,3,height,height)} if platform=='x5' else {'y':(1,height,height,1),'uv':(1,height//2,height//2,2)}
                    channels={'detect':[80,4],'seg':[80,4,32],'pose':[1,4,51],'obb':[15,4,1]}
                    outputs={'logits':(1,1000)} if task=='cls' else {f'{g}-{c}':(1,g,g,c) for g in (2,8,4) for c in reversed(channels[task])}
                    if task=='seg':outputs['proto']=(1,16,16,32)
                    model=types.SimpleNamespace(model_names=['m'],input_names={'m':names},input_shapes={'m':shapes},
                        input_dtypes={'m':{name:'NV12' if platform=='x5' else 'U8' for name in names}},
                        output_names={'m':list(outputs)},output_shapes={'m':outputs},
                        output_dtypes={'m':{name:'F32' for name in outputs}})
                    Model,Config=get_task_types(profile,'yolo26',task)
                    sdk=types.SimpleNamespace(HB_HBMRuntime=lambda _:model)
                    if task=='detect':
                        # The detector's explicit host seam still exercises
                        # full metadata binding; it does not claim a host board.
                        runtime=Model(Config('stub',platform=profile),runtime_loader=lambda:sdk)
                    else:
                        with patch('yolo_runtime.load_hbm_runtime',return_value=sdk):
                            runtime=Model(Config('stub',platform=profile))
                    bound=runtime.pre_process(np.zeros((32,48,3),np.uint8))['m']
                    self.assertEqual(list(bound),names)
                    self.assertEqual(sum(t.size for t in bound.values()),height*height*3//2)

    def test_export_platform_defaults_reach_export(self):
        for task in ('detect','cls','seg','pose','obb'):
            path=S/f'conversion/yolo26/export_yolo26_{task}_bpu.py'
            spec=importlib.util.spec_from_file_location('export_test_'+task,path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            ul=types.ModuleType('ultralytics');head=types.ModuleType('ultralytics.nn.modules')
            for name in ('Detect','Classify','Segment','Segment26','Pose','OBB','OBB26'):setattr(head,name,type(name,(),{}))
            model=MagicMock();model.export.return_value='output.onnx';ul.YOLO=lambda _:model
            torch=types.ModuleType('torch'); nn=types.ModuleType('torch.nn'); torch.nn=nn
            if task=='obb':model.model.model=[head.OBB26()]
            modules={'torch':torch,'torch.nn':nn,'ultralytics':ul,'ultralytics.nn.modules':head}
            if task=='cls':module.convert_linear_to_conv=lambda _:False
            for platform,opset,simplify in [('x5',11,True),('s600',19,False)]:
                with self.subTest(task=task,platform=platform),patch.dict(sys.modules,modules),patch.object(sys,'argv',['export','--weights','mock.pt','--output','output.onnx','--platform',platform]):
                    module.main()
                    self.assertEqual(model.export.call_args.kwargs['opset'],opset)
                    self.assertEqual(model.export.call_args.kwargs['simplify'],simplify)
            with patch.dict(sys.modules,modules),patch.object(sys,'argv',['export','--pt=mock.pt','--output','output.onnx','--platform','s600','--opset','17','--simplify','1']):
                module.main()
                self.assertEqual(model.export.call_args.kwargs['opset'],17)
                self.assertTrue(model.export.call_args.kwargs['simplify'])
                model.export.side_effect=RuntimeError('test exporter failure')
                with self.assertRaises(RuntimeError):module.main()

    def test_all_task_renderers(self):
        import main
        from yolo_platform import resolve_platform
        boxes=np.array([[2,2,20,20]],np.float32)
        scores=np.array([.9]); ids=np.array([0])
        results={'detect':(boxes,scores,ids), 'seg':(boxes,scores,ids,[np.ones((18,18),np.uint8)]),
                 'pose':(boxes,scores,ids,np.ones((1,17,2))*10,np.ones((1,17,1))),
                 'obb':[{'rrect':(16,16,10,8,.2),'score':.9,'id':0}], 'cls':[(0,.9)]}
        with tempfile.TemporaryDirectory() as directory:
            for task,result in results.items():
                with self.subTest(task=task):
                    args=main.build_parser().parse_args(['--family','yolo26','--task',task,'--platform','x5','--img-save-path',str(Path(directory)/(task+'.jpg'))])
                    model=MagicMock();model.predict.return_value=result
                    with patch('yolo_dispatch.create_runtime_model',return_value=model),patch('rdk_yolo_utils.file_io.load_image',return_value=np.zeros((32,32,3),np.uint8)),patch('rdk_yolo_utils.inspect.print_model_info'):
                        main.run_inference(resolve_platform('x5'),args,['test'])
                    if task!='cls':self.assertTrue(Path(args.img_save_path).is_file())

    def test_legacy_cli_and_help(self):
        def run(path,*args):
            result=subprocess.run([sys.executable,str(path),*args],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            return result.stdout
        for platform in ('x5','s'):
            old=R/f'platforms/{platform}/samples/vision/ultralytics_yolo26'
            for task in ('detect','cls','seg','pose','obb'):
                with self.subTest(platform=platform,task=task):
                    target='x5' if platform=='x5' else 's600'
                    output=run(old/'runtime/python/main.py','--platform',target,'--task',task,'--dry-run')
                    self.assertIn('yolo26',output)
                    run(old/f'evaluator/eval_yolo26_{"det" if task=="detect" else task}.py','--help')
                    run(old/f'conversion/onnx_export/export_yolo26_{task}_bpu.py','--help')
            run(old/'conversion/mapper.py','--help')

    def test_legacy_evaluator_defaults_parse(self):
        import runpy
        execute=runpy.run_path
        sys.path.insert(0,str(S/'evaluator'))
        for platform in ('x5','s'):
            for task in ('det','cls','seg','pose','obb'):
                with self.subTest(platform=platform,task=task):
                    old=R/f'platforms/{platform}/samples/vision/ultralytics_yolo26/evaluator/eval_yolo26_{task}.py'
                    def capture(*args,**kwargs):return list(sys.argv[1:])
                    with patch.object(sys,'argv',[str(old),'--model-path','custom.bin']),patch('runpy.run_path',side_effect=capture) as forward:
                        execute(str(old),run_name='__main__')
                        argv=list(sys.argv[1:])
                    evaluator=importlib.import_module('eval_yolo_'+task)
                    args=evaluator.build_parser().parse_args(argv)
                    self.assertEqual(args.family,'yolo26')
                    self.assertTrue(args.image_dir)

    def test_mask_inverse_letterbox_at_non640_size(self):
        from rdk_yolo_utils.postprocess import process_mask
        proto=np.full((32,16,16),10,np.float32)
        coefficients=np.ones((1,32),np.float32)
        # 64x32 original image occupies y=16:48 in a 64-square letterbox.
        mask=process_mask(proto,coefficients,np.array([[0,16,64,48]],np.float32),(32,64),64,64,1)
        self.assertEqual(mask.shape,(1,32,64))
        self.assertTrue(mask.all())

    def test_dispatch_uses_ltrb_not_dfl(self):
        from yolo_dispatch import get_task_types
        from yolo_platform import resolve_platform
        from yolo_detect import YoloDetect
        from yolo26_det import YOLO26Detect
        profile = resolve_platform('x5')
        self.assertIs(get_task_types(profile, 'yolov8', 'detect')[0], YoloDetect)
        self.assertIs(get_task_types(profile, 'yolo26', 'detect')[0], YOLO26Detect)

    def test_all_100_assets_match_catalog(self):
        from yolo_assets import model_url
        from yolo_platform import resolve_platform
        data=json.loads((R/'tools/catalog-publisher/dist/catalog.json').read_text(encoding='utf-8'))
        published={a['url'] for m in data['models'] for a in m.get('assets',[]) if a.get('url')}
        urls=set()
        for platform in ('x5','s100','s100p','s600'):
            for task in ('detect','cls','seg','pose','obb'):
                for size in 'nsmlx':
                    url=model_url(resolve_platform(platform),'yolo26',task,size)
                    self.assertTrue(url in published,url)
                    if task=='cls':self.assertIn('_224x224_',url)
                    urls.add(url)
        self.assertEqual(len(urls),100)

    def test_output_order_uses_geometry_not_runtime_list_order(self):
        from yolo26_common import ordered_outputs
        shapes={}
        for grid in (2,8,4):
            for channel in (4,80,32):shapes[f'{grid}-{channel}']=(1,grid,grid,channel)
        shapes['prototype']=(1,16,16,32)
        expected=[f'{g}-{c}' for g in (8,4,2) for c in (80,4,32)]+['prototype']
        self.assertEqual(ordered_outputs(list(shapes),shapes,64,[8,16,32],'seg',80),expected)

    def test_obb_angles_are_radians_on_all_platforms(self):
        from yolo26_obb import YOLO26OBB, YOLO26OBBConfig
        from yolo_platform import resolve_platform
        shapes = {f'{g}-{c}': (1, g, g, c)
                  for g in (20, 80, 40) for c in (1, 4, 15)}
        for platform in ('x5', 's100', 's100p', 's600'):
            with self.subTest(platform=platform):
                inputs = {'image': (1, 3, 640, 640)} if platform == 'x5' else {
                    'y': (1, 640, 640, 1), 'uv': (1, 320, 320, 2)}
                model = types.SimpleNamespace(
                    model_names=['m'], input_names={'m': list(inputs)}, input_shapes={'m': inputs},
                    input_dtypes={'m': {n: 'NV12' if platform == 'x5' else 'U8' for n in inputs}},
                    output_names={'m': list(shapes)}, output_shapes={'m': shapes})
                sdk = types.SimpleNamespace(HB_HBMRuntime=lambda _: model)
                with patch('yolo_runtime.load_hbm_runtime', return_value=sdk):
                    runtime = YOLO26OBB(YOLO26OBBConfig('stub', platform=resolve_platform(platform)))
                outputs = {name: np.zeros(shape, np.float32) for name, shape in shapes.items()}
                for g in (20, 40, 80):
                    outputs[f'{g}-15'].fill(-20)
                outputs['20-15'][0, 7, 15, 9] = 3
                outputs['20-4'][0, 7, 15] = [1, 1, 3, 1]
                outputs['20-1'][0, 7, 15, 0] = np.pi / 2
                result = runtime.post_process({'m': outputs}, 640, 640)[0]
                np.testing.assert_allclose(result['rrect'][:4], [496, 272, 128, 64], atol=1e-4)
                self.assertAlmostEqual(np.cos(2 * result['rrect'][4]), -1, places=6)
                self.assertEqual(result['id'], 9)

    def test_obb_default_labels_and_custom_override(self):
        from main import load_labels
        labels = load_labels(types.SimpleNamespace(label_file=''), 'obb')
        self.assertEqual(labels[9:11], ['large-vehicle', 'small-vehicle'])
        self.assertEqual(labels[1], 'ship')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.txt'
            path.write_text('custom-object\n', encoding='utf-8')
            self.assertEqual(load_labels(types.SimpleNamespace(label_file=str(path)), 'obb'), ['custom-object'])

    def test_seg_mask_alignment_for_portrait_and_landscape(self):
        from rdk_yolo_utils.postprocess import process_mask
        proto = np.full((32, 16, 16), 10, np.float32)
        coefficients = np.ones((1, 32), np.float32)
        for shape in ((64, 32), (32, 64)):
            with self.subTest(shape=shape):
                masks = process_mask(proto, coefficients, np.array([[28, 28, 44, 44]], np.float32), shape, 64, 64, 1)
                self.assertEqual(masks.shape, (1, *shape))
                y, x = (40, 24) if shape == (64, 32) else (24, 40)
                self.assertTrue(masks[0, y, x])
                self.assertFalse(masks[0, 0, 0])

    def test_dfl_output_rejected_as_ltrb(self):
        from yolo26_common import ordered_outputs
        shapes={f'{g}-{c}':(1,g,g,c) for g in (8,4,2) for c in (80,64)}
        with self.assertRaises(ValueError):ordered_outputs(list(shapes),shapes,64,[8,16,32],'detect',80)

    def test_640_output_groups_match_stride_for_pose_seg_and_obb(self):
        from yolo26_common import ordered_outputs
        for task, channels, classes in (
            ('pose', (1, 4, 51), 1),
            ('seg', (80, 4, 32), 80),
            ('obb', (15, 4, 1), 15),
        ):
            with self.subTest(task=task):
                shapes = {f'{g}-{c}': (1, g, g, c)
                          for g in (20, 40, 80) for c in reversed(channels)}
                expected = [f'{g}-{c}' for g in (80, 40, 20) for c in channels]
                if task == 'seg':
                    shapes['prototype'] = (1, 160, 160, 32)
                    expected.append('prototype')
                for names in (list(shapes), list(reversed(shapes))):
                    self.assertEqual(ordered_outputs(
                        names, shapes, 640, [8, 16, 32], task, classes), expected)

    def test_pose_right_side_coordinates_survive_output_reordering(self):
        from yolo26_pose import YOLO26Pose, YOLO26PoseConfig
        from yolo_platform import resolve_platform
        # Inject only the board loader; use real metadata binding and decoding.
        shapes = {f'{g}-{c}': (1, g, g, c)
                  for g in (20, 80, 40) for c in (51, 4, 1)}
        model = types.SimpleNamespace(
            model_names=['m'], input_names={'m': ['image']},
            input_shapes={'m': {'image': (1, 3, 640, 640)}},
            input_dtypes={'m': {'image': 'NV12'}},
            output_names={'m': list(shapes)}, output_shapes={'m': shapes})
        sdk = types.SimpleNamespace(HB_HBMRuntime=lambda _: model)
        with patch('yolo_runtime.load_hbm_runtime', return_value=sdk):
            runtime = YOLO26Pose(YOLO26PoseConfig('stub', platform=resolve_platform('x5')))
        for grid, row, col, box, point in (
            (80, 30, 60, [476, 236, 492, 252], [484, 244]),
            (40, 15, 30, [472, 232, 504, 264], [488, 248]),
            (20, 7, 15, [464, 208, 528, 272], [496, 240]),
        ):
            with self.subTest(grid=grid):
                outputs = {name: np.zeros(shape, np.float32) for name, shape in shapes.items()}
                for g in (20, 40, 80):
                    outputs[f'{g}-1'].fill(-20)
                outputs[f'{grid}-1'][0, row, col, 0] = 3
                outputs[f'{grid}-4'][0, row, col] = 1
                boxes, scores, ids, xy, confidence = runtime.post_process({'m': outputs}, 640, 640)
                np.testing.assert_allclose(boxes, [box])
                np.testing.assert_allclose(xy, [[point] * 17])
                np.testing.assert_allclose(scores, [0.95257413])
                np.testing.assert_allclose(confidence, .5)

if __name__=='__main__':unittest.main()
