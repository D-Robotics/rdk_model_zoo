"""Host tests of real legacy/unified SAM evidence capture, never board claims."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from samples._shared.sam_binding import resolve_selection
from samples._shared.tests.test_sam_binding import FakeRuntime,metadata
from samples._shared.sam_evaluator import run_comparison,build_parser


class SAMEvaluatorTests(unittest.TestCase):
    def test_all_four_source_paths_with_full_evidence(self):
        for sample in ('efficient_sam','mobile_sam'):
            for target in ('x5','s100'):
                with self.subTest(sample=sample,target=target), tempfile.TemporaryDirectory() as td:
                    tmp=Path(td)
                    image=np.arange(17*31*3,dtype=np.uint8).reshape(17,31,3)
                    img=tmp/'input.raw'; img.write_bytes(image.tobytes())
                    base=resolve_selection(sample,target)
                    ep=tmp/'encoder.fixture';ep.write_bytes(b'host-fixture-encoder')
                    dp=tmp/'decoder.fixture';dp.write_bytes(b'host-fixture-decoder')
                    selection=resolve_selection(sample,target,encoder_model_path=ep,decoder_model_path=dp,
                        encoder_asset_id=base.encoder_asset.reference,decoder_asset_id=base.decoder_asset.reference)
                    runtimes=[]
                    def factory(path):
                        stage='encoder' if Path(path)==ep else 'decoder'
                        fake=FakeRuntime(metadata(sample,stage,target))
                        if stage=='decoder': fake.outputs['iou_predictions'].reshape(-1)[:]=[0.1,0.9,0.2]
                        runtimes.append(fake)
                        return fake
                    with patch('samples._shared.sam_evaluator.require_execution_target',return_value=target) as gate:
                        summary=run_comparison(selection,image,img,tmp/'evidence',runtime_factory=factory)
                    gate.assert_called_once_with(target)
                    self.assertTrue(summary['passed'])
                    self.assertEqual(len(runtimes),4)
                    self.assertTrue(summary['checks']['inputs_equal'])
                    self.assertEqual(summary['mask_changed_pixels'],0)
                    stored=json.loads((tmp/'evidence/comparison.json').read_text())
                    self.assertTrue(stored['passed'])
                    self.assertEqual(stored['sample'],sample)
                    self.assertEqual(stored['target'],target)
                    self.assertEqual(len(stored['artifacts']),2)
                    self.assertGreaterEqual(len(list((tmp/'evidence').glob('*.npy'))),14)
                    self.assertGreaterEqual(len(stored['code_sha256']),8)
                    with self.assertRaises(FileExistsError):
                        run_comparison(selection,image,img,tmp/'evidence',runtime_factory=factory)

    def test_runtime_mask_regression_retains_failed_full_capture(self):
        with tempfile.TemporaryDirectory() as td:
            tmp=Path(td); img=tmp/'input.fixture';img.write_bytes(b'image')
            ep=tmp/'encoder.fixture';ep.write_bytes(b'encoder')
            dp=tmp/'decoder.fixture';dp.write_bytes(b'decoder')
            base=resolve_selection('efficient_sam','s100')
            selection=resolve_selection('efficient_sam','s100',encoder_model_path=ep,decoder_model_path=dp,
                encoder_asset_id=base.encoder_asset.reference,decoder_asset_id=base.decoder_asset.reference)
            calls=[]
            def factory(path):
                stage='encoder' if Path(path)==ep else 'decoder'
                runtime=FakeRuntime(metadata('efficient_sam',stage,'s100'))
                calls.append(runtime)
                if stage=='decoder':
                    runtime.outputs['iou_predictions'].reshape(-1)[:]=[0.9,0.1,0.2]
                    if len(calls)==4:runtime.outputs['low_res_masks'][0,0,0,0]=-3
                return runtime
            with patch('samples._shared.sam_evaluator.require_execution_target',return_value='s100'):
                actual=run_comparison(selection,np.zeros((3,5,3),dtype=np.uint8),img,tmp/'evidence',runtime_factory=factory)
            self.assertFalse(actual['passed']);self.assertFalse(actual['checks']['mask_equal'])
            stored=json.loads((tmp/'evidence/comparison.json').read_text())
            self.assertEqual(stored['return_code'],1);self.assertFalse(stored['passed'])
            self.assertEqual(len(stored['arrays']),14)
            self.assertGreater(stored['mask_changed_pixels'],0)
            self.assertEqual(set(stored['metadata']),{'legacy','unified'})

    def test_missing_model_error_keeps_requested_evidence_record(self):
        with tempfile.TemporaryDirectory() as td:
            tmp=Path(td); img=tmp/'image.fixture'; img.write_bytes(b'fixture')
            base=resolve_selection('efficient_sam','s100')
            selected=resolve_selection('efficient_sam','s100',encoder_model_path=tmp/'missing.fixture',
                encoder_asset_id=base.encoder_asset.reference)
            with patch('samples._shared.sam_evaluator.require_execution_target',return_value='s100'):
                with self.assertRaises(FileNotFoundError):
                    run_comparison(selected,np.zeros((2,3,3),dtype=np.uint8),img,tmp/'evidence')
            evidence=json.loads((tmp/'evidence/comparison.json').read_text())
            self.assertFalse(evidence['passed']);self.assertEqual(evidence['return_code'],2)
            self.assertEqual(evidence['error']['type'],'FileNotFoundError')

    def test_dtype_or_mask_mismatch_is_saved_and_fails(self):
        from samples._shared.sam_evaluator import compare_records
        z=np.zeros((1,3,2,2),dtype=np.float32)
        record={'inputs':{'encoder':{'batched_images':z}},
                'outputs':{'encoder':{'image_embeddings':z}},
                'result':{'mask':np.zeros((512,512),dtype=bool),'iou':0.9,'mask_index':1,'low_res_masks':z}}
        other={'inputs':{'encoder':{'batched_images':z.astype(np.float64)}},
                'outputs':record['outputs'],'result':record['result']}
        result=compare_records(record,other)
        self.assertFalse(result['passed']); self.assertFalse(result['checks']['inputs_equal'])
        mask=record['result']['mask'].copy();mask[2,4]=True
        other={'inputs':record['inputs'],'outputs':record['outputs'],'result':{**record['result'],'mask':mask}}
        result=compare_records(record,other)
        self.assertFalse(result['passed']);self.assertEqual(result['mask_changed_pixels'],1)
        other={'inputs':record['inputs'],'outputs':{'encoder':{'image_embeddings':z.reshape(-1)}},'result':record['result']}
        self.assertFalse(compare_records(record,other)['passed'])

    def test_parser_defaults_and_no_implicit_target(self):
        for sample in ('efficient_sam','mobile_sam'):
            parser=build_parser(sample)
            a=parser.parse_args(['--target','s100','--output-dir','/tmp/new-evidence'])
            self.assertEqual(a.priority,0);self.assertIsNone(a.bpu_cores)
            self.assertIn('dogs.jpg',str(a.test_img))

if __name__=='__main__':unittest.main()
