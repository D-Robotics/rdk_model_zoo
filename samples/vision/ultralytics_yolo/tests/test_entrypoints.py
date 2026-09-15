"""Public command and tensor contracts; no board or network required."""
from pathlib import Path
import sys, subprocess, tempfile, unittest, importlib.util, os
from unittest.mock import patch
import numpy as np
S=Path(__file__).resolve().parents[1];R=S.parents[2]
sys.path.insert(0,str(S/'runtime/python'))
from yolo_platform import resolve_platform
from yolo_input import Nv12InputAdapter, UnsupportedInputError
from yolo_runtime import default_resize_type, default_nms_thres
def command(path,*args):
    result=subprocess.run([sys.executable,str(path),*args],cwd=tempfile.gettempdir(),capture_output=True,text=True,timeout=25)
    return result
class Entrypoints(unittest.TestCase):
    def test_help_without_board(self):
        paths=list((S/'evaluator').glob('eval_*.py'))+[S/'runtime/python/main.py',S/'runtime/python/yolo_download.py',S/'conversion/mapper.py',S/'conversion/export_monkey_patch.py']
        paths=[p for p in paths if p.name!='eval_common.py']
        for tree in ('x5','s'):
            old=R/f'platforms/{tree}/samples/vision/ultralytics_yolo'
            paths += list((old/'evaluator').glob('*.py'))+list((old/'conversion').glob('*.py'))+[old/'runtime/python/main.py']
        for path in paths:
            with self.subTest(path=path):
                r=command(path,'--help');self.assertEqual(r.returncode,0,r.stderr);self.assertIn('usage:',r.stdout)
    def test_custom_model_does_not_download_and_selects_v10(self):
        r=command(S/'runtime/python/main.py','--platform','s600','--model-path','yolov10n_custom.hbm','--dry-run')
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn('family          : yolov10',r.stdout);self.assertNotIn('https://',r.stdout)
    def test_defaults(self):
        self.assertEqual(default_resize_type(resolve_platform('x5'),'cls'),1)
        self.assertEqual(default_resize_type(resolve_platform('s600'),'cls'),0)
        self.assertEqual(default_nms_thres(resolve_platform('x5'),'detect'),.7)
        self.assertEqual(default_nms_thres(resolve_platform('s100'),'detect'),.45)
    def test_planes(self):
        y=np.arange(16,dtype=np.uint8);uv=np.arange(8,dtype=np.uint8)
        a=Nv12InputAdapter.from_metadata(resolve_platform('x5'),'m',['image'],{'image':(1,3,4,4)})
        np.testing.assert_array_equal(a.build(y,uv)['m']['image'],np.concatenate([y,uv]))
        b=Nv12InputAdapter.from_metadata(resolve_platform('s600'),'m',['y','uv'],{'y':(1,4,4,1),'uv':(1,2,2,2)})
        result=b.build(y,uv)['m'];self.assertEqual(result['y'].shape,(1,4,4,1));self.assertEqual(result['uv'].shape,(1,2,2,2))
    def test_input_count_mismatch(self):
        with self.assertRaises(UnsupportedInputError):
            Nv12InputAdapter.from_metadata(resolve_platform('s100'),'m',['image'],{'image':(1,3,640,640)})
    def test_conversion_conflict(self):
        r=command(S/'conversion/mapper.py','--platform','s600','--march','nash-e')
        self.assertEqual(r.returncode,2);self.assertIn('different architectures',r.stderr)
    def test_s_import_is_not_circular(self):
        old=R/'platforms/s/samples/vision/ultralytics_yolo/runtime/python'
        code='import sys;sys.path.insert(0,sys.argv[1]);import yolo_detect,yolo_seg,yolo_pose,yolo_cls,yolo_v10detect;print("ok")'
        r=subprocess.run([sys.executable,'-c',code,str(old)],cwd=tempfile.gettempdir(),capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
    def test_pose_legacy_returns_three(self):
        p=R/'platforms/x5/samples/vision/ultralytics_yolo/runtime/python/ultralytics_yolo_pose.py'
        spec=importlib.util.spec_from_file_location('legacy_pose_test',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
        result=(np.zeros((0,4)),np.zeros(0),np.zeros(0),np.zeros((0,17,2)),np.zeros((0,17,1)))
        with patch.object(m._BaseModel,'post_process',return_value=result):
            out=m.UltralyticsYOLOPose.__new__(m.UltralyticsYOLOPose).post_process(None)
        self.assertEqual(len(out),3);self.assertEqual(out[2].shape,(0,17,3))
if __name__=='__main__':unittest.main()
