"""Runner purity, identity and SDK-free user entrypoints."""
import subprocess,sys,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from samples.vision.yolov5.tests.test_yolov5 import FakeRuntime
from samples.vision.yolov5.runtime.python.model_binding import resolve_selection
from samples.vision.yolov5.runtime.python.model_runner import RuntimeModelRunner
ROOT=Path(__file__).resolve().parents[4]

class RunnerCLITests(unittest.TestCase):
    def test_identity_before_sdk_import(self):
        with patch('samples._shared.platforms.require_execution_target',side_effect=ValueError('Target mismatch')) as gate:
            with patch('samples._shared.model_runner._default_runtime_factory') as factory:
                with self.assertRaisesRegex(ValueError,'Target mismatch'):RuntimeModelRunner(resolve_selection('s100')).load()
                gate.assert_called_once_with('s100');factory.assert_not_called()

    def test_runner_named_container_native_output_and_schedule(self):
        for target in ('x5','s100'):
            fake=FakeRuntime(target);runner=RuntimeModelRunner(resolve_selection(target),runtime_factory=lambda _:fake);binding=runner.load()
            from samples.vision.yolov5.runtime.python.detection import YOLOv5Task
            task=YOLOv5Task(runner,binding);prepared=task.pre_process(np.zeros((17,23,3),np.uint8));raw=task.forward(prepared.tensors)
            self.assertEqual(set(fake.calls[-1]),{'detector'})
            for k in raw:self.assertIs(raw[k],fake.outputs[k])
            runner.set_scheduling_params(priority=7,bpu_cores=[0]);self.assertEqual(fake.schedule,{'priority':{'detector':7},'bpu_cores':{'detector':[0]}})

    def test_all_safe_cli_modes_from_arbitrary_cwd(self):
        script=ROOT/'samples/vision/yolov5/runtime/python/main.py'
        for args in [('--help',),('--list-models',),('--dry-run','--target','x5'),('--dry-run','--target','s100'),('--dry-run','--target','s600')]:
            p=subprocess.run([sys.executable,str(script),*args],cwd='/tmp',text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr)
        for args in [('--dry-run',),('--dry-run','--target','s100p'),('--dry-run','--target','x5','--score-thres','nan')]:
            p=subprocess.run([sys.executable,str(script),*args],cwd='/tmp',text=True,capture_output=True)
            self.assertEqual(p.returncode,2,p.stdout+p.stderr)

    def test_model_downloader_only_fetches_selected_manifest_row(self):
        from samples.vision.yolov5.model import download
        with tempfile.TemporaryDirectory() as td,patch.object(download,'download_asset',return_value='fixture-hash') as fetch:
            asset,path,digest=download.download_target('x5',td,variant='s-v2.0')
            self.assertEqual(path,Path(td)/asset.filename);self.assertIn('s_tag_v2.0',asset.filename)
            fetch.assert_called_once_with(asset,path)

if __name__=='__main__':unittest.main()
