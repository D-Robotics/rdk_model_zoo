"""Public command and tensor contracts; no board or network required."""
from pathlib import Path
import ast, sys, subprocess, tempfile, unittest, importlib.util, os
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
    def test_main_visibly_constructs_and_predicts(self):
        # The readable-example ruling: main() itself must construct the
        # dispatched task model and call predict, not hide the flow behind
        # a helper. Guard the structure so a refactor cannot silently move
        # it out again.
        tree=ast.parse((S/'runtime/python/main.py').read_text(encoding='utf-8'))
        main_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        calls=[n for n in ast.walk(main_fn) if isinstance(n,ast.Call)]
        constructed=any(
            isinstance(c.func,ast.Name) and c.func.id=='Model'
            for c in calls)
        predicts=[c for c in calls
                  if isinstance(c.func,ast.Attribute) and c.func.attr=='predict'
                  and isinstance(c.func.value,ast.Name) and c.func.value.id=='model']
        self.assertTrue(constructed,'main() must visibly call Model(config)')
        self.assertTrue(predicts,'main() must call model.predict itself')
    def test_help_without_board(self):
        paths=list((S/'evaluator').glob('eval_*.py'))+[S/'runtime/python/main.py',S/'runtime/python/yolo_download.py',S/'conversion/mapper.py',S/'conversion/export_monkey_patch.py']
        paths=[p for p in paths if p.name!='eval_common.py']
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
if __name__=='__main__':unittest.main()
