"""SDK-free commands and real CPU source/unified evidence with fake detector SDK."""
import importlib,json,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from samples.vision.bytetrack.runtime.python.model_binding import resolve_selection
from samples.vision.bytetrack.evaluator.capture import capture_frames
from samples.vision.bytetrack.evaluator.compare import compare_captures
from samples.vision.yolov5.tests.test_yolov5 import FakeRuntime
ROOT=Path(__file__).resolve().parents[4]


class _BoardQuantParams:
    """Mimics hbm_runtime.QuantParams: attributes read, any copy refuses (X5 board evidence 2026-09-24)."""
    def __init__(self,quant_type,scale,zero_point,axis):
        self.quant_type=types.SimpleNamespace(name=quant_type);self.scale=scale;self.zero_point=zero_point;self.axis=axis
    def __deepcopy__(self,memo):raise TypeError("cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object")
    def __copy__(self):raise TypeError("cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object")


class EntryEvidenceTests(unittest.TestCase):
    def test_safe_cli_and_exact_manual_path_identity(self):
        script=ROOT/'samples/vision/bytetrack/runtime/python/main.py'
        for args in [('--help',),('--list-models',),('--dry-run','--target','s100p')]:
            p=subprocess.run([sys.executable,str(script),*args],cwd='/tmp',text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        for args in [('--dry-run',),('--dry-run','--target','x5'),('--dry-run','--target','s100','--track-thresh','nan')]:
            p=subprocess.run([sys.executable,str(script),*args],cwd='/tmp',text=True,capture_output=True);self.assertEqual(p.returncode,2,p.stdout+p.stderr)
        with self.assertRaises(ValueError):resolve_selection('s100',model_path='/tmp/model')
        with self.assertRaises(ValueError):resolve_selection('s100p',asset_id=resolve_selection('s100').asset.reference)

    def test_download_and_shell_are_explicit_and_offline_fixtures(self):
        from samples.vision.bytetrack.model import download
        with tempfile.TemporaryDirectory() as td,patch.object(download,'download_asset',return_value='fixture') as fetch:
            asset,path,_=download.download_target('s100p',td);self.assertEqual(asset.filename,'s100p/yolov5x_672x672_nv12.hbm');fetch.assert_called_once_with(asset,path)
            binary=Path(td)/'python3';log=Path(td)/'args.txt';binary.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n');binary.chmod(0o755)
            import os
            env={**os.environ,'PATH':td+':'+os.environ['PATH'],'CAPTURE_ARGS':str(log)}
            script=ROOT/'samples/vision/bytetrack/model/download_model.sh'
            p=subprocess.run(['bash',str(script),'--target','s100p'],env=env,cwd='/tmp',text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(log.read_text().splitlines()[1:],['--target','s100p'])

    def test_complete_stream_capture_and_difference_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            tmp=Path(td);mp=tmp/'model.fixture';mp.write_bytes(b'model');vp=tmp/'video.fixture';vp.write_bytes(b'video')
            base=resolve_selection('s100p');selected=resolve_selection('s100p',model_path=mp,asset_id=base.asset.reference)
            images=[np.zeros((97,151,3),np.uint8) for _ in range(3)]
            from samples.vision.bytetrack.runtime.python.tracker_backend.basetrack import BaseTrack
            oldpath=list(sys.path);sys.path.insert(0,str(ROOT/'platforms/s/samples/vision/bytetrack/3rdparty'))
            try:LegacyBase=importlib.import_module('tracker.basetrack').BaseTrack
            finally:sys.path[:]=oldpath
            # Production uses fresh CLI subprocesses. Fixtures isolate those counters explicitly.
            def runtime(_):
                value=FakeRuntime('s100p','int8')
                for output in value.outputs.values():
                    output.fill(-80)
                    # A valid person in the image interior, outside letterbox padding.
                    h,w=output.shape[1:3]
                    a=output[0,h//2,w//2,:85];a[:4]=0;a[4]=60;a[5]=50
                return value
            with patch.object(BaseTrack,'_count',0),patch.object(LegacyBase,'_count',0),patch('samples.vision.bytetrack.evaluator.capture.require_execution_target',return_value='s100p'):
                for side in ('legacy','unified'):
                    capture_frames(selected,iter(images),tmp/side,side=side,video_path=vp,runtime_factory=runtime)
            result=compare_captures(tmp/'legacy',tmp/'unified');self.assertTrue(result['passed'],result)
            data=tmp/'unified/00001_input_0.npy'; original=data.read_bytes()
            array=np.load(data);array.reshape(-1)[0]^=1;np.save(data,array)
            self.assertFalse(compare_captures(tmp/'legacy',tmp/'unified')['passed'])
            data.write_bytes(original)
            record=json.loads((tmp/'unified/capture.json').read_text());self.assertEqual(len(record['frames']),3);self.assertEqual(len(list((tmp/'unified').glob('*.npy'))),18)
            record['frames'][0]['tracks'][0]['track_id']+=1;(tmp/'unified/capture.json').write_text(json.dumps(record))
            self.assertFalse(compare_captures(tmp/'legacy',tmp/'unified')['passed'])

    def test_capture_metadata_survives_copy_hostile_board_quant_params(self):
        """The old asdict() metadata snapshot raised TypeError on the real board."""
        with tempfile.TemporaryDirectory() as td:
            tmp=Path(td);mp=tmp/'model.fixture';mp.write_bytes(b'model');vp=tmp/'video.fixture';vp.write_bytes(b'video')
            base=resolve_selection('s100p');selected=resolve_selection('s100p',model_path=mp,asset_id=base.asset.reference)
            def runtime(_):
                value=FakeRuntime('s100p','int8')
                for output in value.outputs.values():
                    output.fill(-80)
                    h,w=output.shape[1:3]
                    a=output[0,h//2,w//2,:85];a[:4]=0;a[4]=60;a[5]=50
                quant=_BoardQuantParams('SCALE',np.linspace(.08,.12,255,dtype=np.float32),np.zeros(255,dtype=np.int32),3)
                value.output_quants={'detector':{n:quant for n in value.facts['output_names']}}
                return value
            with patch('samples.vision.bytetrack.evaluator.capture.require_execution_target',return_value='s100p'):
                summary=capture_frames(selected,iter([np.zeros((97,151,3),np.uint8)]),tmp/'unified',side='unified',video_path=vp,runtime_factory=runtime)
            self.assertEqual(summary['return_code'],0,summary.get('error'))
            quants=summary['metadata']['output_quants'];self.assertEqual(set(quants),{'small','medium','large'})
            for entry in quants.values():
                self.assertEqual(entry['quant_type'],'SCALE');self.assertEqual(len(entry['scale']),255)
                self.assertEqual(entry['zero_point'],[0]*255);self.assertEqual(entry['axis'],3)
                self.assertEqual(entry['scale'][0],np.float32(.08).item())
            saved=json.loads((tmp/'unified/capture.json').read_text())
            self.assertEqual(saved['metadata']['output_quants']['small']['scale'][0],np.float32(.08).item())

    def test_missing_model_keeps_error_capture(self):
        with tempfile.TemporaryDirectory() as td:
            tmp=Path(td);vp=tmp/'video';vp.write_bytes(b'video');base=resolve_selection('s100')
            selected=resolve_selection('s100',model_path=tmp/'missing',asset_id=base.asset.reference)
            with patch('samples.vision.bytetrack.evaluator.capture.require_execution_target',return_value='s100'):
                with self.assertRaises(FileNotFoundError):capture_frames(selected,[],tmp/'evidence',side='unified',video_path=vp,runtime_factory=lambda _:FakeRuntime('s100'))
            record=json.loads((tmp/'evidence/capture.json').read_text());self.assertEqual(record['return_code'],2);self.assertIn('error',record)

    def test_source_nan_state_is_recorded_as_failure(self):
        with tempfile.TemporaryDirectory() as td:
            tmp=Path(td);mp=tmp/'model';mp.write_bytes(b'model');vp=tmp/'video';vp.write_bytes(b'video')
            base=resolve_selection('s100');selection=resolve_selection('s100',model_path=mp,asset_id=base.asset.reference)
            with patch('samples.vision.bytetrack.evaluator.capture.require_execution_target',return_value='s100'):
                with self.assertRaisesRegex(ValueError,'non-finite state'):
                    capture_frames(selection,[np.zeros((97,151,3),np.uint8)],tmp/'legacy',side='legacy',video_path=vp,runtime_factory=lambda _:FakeRuntime('s100','int8'))
            record=json.loads((tmp/'legacy/capture.json').read_text())
            self.assertEqual(record['return_code'],2);self.assertIn('non-finite',record['error']['message'])
            self.assertEqual(len(record['frames'][0]['outputs']),3)

if __name__=='__main__':unittest.main()
