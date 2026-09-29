"""Exercise actual source/unified evidence paths using an injected offline SDK."""
import json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from samples.vision.yolov5.tests.test_yolov5 import FakeRuntime
from samples.vision.yolov5.runtime.python.model_binding import resolve_selection
from samples.vision.yolov5.evaluator.compare import run_comparison


class _BoardQuantParams:
    """Mimics hbm_runtime.QuantParams: attributes read, any copy refuses (X5 board evidence 2026-09-24)."""
    def __init__(self,quant_type,scale,zero_point,axis):
        self.quant_type=types.SimpleNamespace(name=quant_type);self.scale=scale;self.zero_point=zero_point;self.axis=axis
    def __deepcopy__(self,memo):raise TypeError("cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object")
    def __copy__(self):raise TypeError("cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object")


class EvaluatorTests(unittest.TestCase):
    def test_source_and_unified_comparison_retains_all_tensors(self):
        for target in ('x5','s100','s600'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as td:
                tmp=Path(td);mp=tmp/'model.fixture';mp.write_bytes(b'model');ip=tmp/'image.fixture';ip.write_bytes(b'image')
                base=resolve_selection(target);selection=resolve_selection(target,asset_id=base.asset.reference,model_path=mp)
                with patch('samples.vision.yolov5.evaluator.compare.require_execution_target',return_value=target):
                    result=run_comparison(selection,np.zeros((97,151,3),np.uint8),ip,tmp/'evidence',runtime_factory=lambda _:FakeRuntime(target,'float32' if target=='x5' else 'int8'))
                self.assertTrue(result['passed']);self.assertEqual(result['return_code'],0)
                saved=json.loads((tmp/'evidence/comparison.json').read_text());self.assertEqual(saved['asset_id'],base.asset.reference)
                self.assertEqual(len(saved['arrays']),14 if target=='x5' else 16)
                self.assertEqual(set(saved['metadata']),{'legacy','unified'});self.assertIsNone(saved['publisher_sha256'])
                self.assertTrue(saved['model_sha256']);self.assertGreater(len(saved['code_sha256']),10)

    def test_metadata_evidence_survives_copy_hostile_board_quant_params(self):
        """The old asdict() evidence snapshot raised TypeError on the real board."""
        def hostile(target,dtype):
            runtime=FakeRuntime(target,dtype)
            quant=_BoardQuantParams('NONE' if dtype=='float32' else 'SCALE',np.linspace(.08,.12,255,dtype=np.float32),np.zeros(255,dtype=np.int32),3)
            runtime.output_quants={'detector':{n:quant for n in runtime.facts['output_names']}}
            return runtime
        for target in ('x5','s100'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as td:
                tmp=Path(td);mp=tmp/'model.fixture';mp.write_bytes(b'model');ip=tmp/'image.fixture';ip.write_bytes(b'image')
                base=resolve_selection(target);selection=resolve_selection(target,asset_id=base.asset.reference,model_path=mp)
                with patch('samples.vision.yolov5.evaluator.compare.require_execution_target',return_value=target):
                    result=run_comparison(selection,np.zeros((97,151,3),np.uint8),ip,tmp/'evidence',runtime_factory=lambda _,t=target:hostile(t,'float32' if t=='x5' else 'int8'))
                self.assertTrue(result['passed']);self.assertEqual(result['return_code'],0)
                saved=json.loads((tmp/'evidence/comparison.json').read_text())
                for side in ('legacy','unified'):
                    quants=saved['metadata'][side]['output_quants']
                    self.assertEqual(set(quants),{'small','medium','large'})
                    for name in quants:
                        entry=quants[name]
                        self.assertEqual(entry['quant_type'],'NONE' if target=='x5' else 'SCALE')
                        self.assertEqual(len(entry['scale']),255);self.assertEqual(entry['zero_point'],[0]*255);self.assertEqual(entry['axis'],3)
                        self.assertEqual(entry['scale'][0],np.float32(.08).item())

    def test_failed_result_and_preload_error_are_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            tmp=Path(td);mp=tmp/'model.fixture';mp.write_bytes(b'model');ip=tmp/'image.fixture';ip.write_bytes(b'image')
            base=resolve_selection('s100');selection=resolve_selection('s100',asset_id=base.asset.reference,model_path=mp);calls=[]
            def factory(_):
                fake=FakeRuntime('s100');calls.append(fake)
                if len(calls)==2:fake.outputs['small'].reshape(-1,85)[3,5]=-8
                return fake
            with patch('samples.vision.yolov5.evaluator.compare.require_execution_target',return_value='s100'):
                result=run_comparison(selection,np.zeros((97,151,3),np.uint8),ip,tmp/'mismatch',runtime_factory=factory)
                self.assertFalse(result['passed']);self.assertEqual(result['return_code'],1)
                mp.unlink()
                with self.assertRaises(FileNotFoundError):run_comparison(selection,np.zeros((97,151,3),np.uint8),ip,tmp/'missing',runtime_factory=factory)
            saved=json.loads((tmp/'missing/comparison.json').read_text());self.assertEqual(saved['return_code'],2);self.assertIn('error',saved)

if __name__=='__main__':unittest.main()
