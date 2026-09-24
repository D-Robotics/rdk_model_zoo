"""Host tests of real legacy/unified SAM evidence capture, never board claims."""
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch
import numpy as np
from samples._shared.sam_binding import resolve_selection
from samples._shared.tests.test_sam_binding import FakeRuntime,metadata
from samples._shared.sam_evaluator import run_comparison,build_parser,main,_digest


class _NoFileDigest:
    """Hide hashlib.file_digest as on real Python 3.10 boards (API added in 3.11)."""
    def __enter__(self):
        self._saved=getattr(hashlib,'file_digest',None)
        if self._saved is not None:del hashlib.file_digest
        return self
    def __exit__(self,*exc):
        if self._saved is not None:hashlib.file_digest=self._saved
        return False


_KNOWN_SHA256={
    b'':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    b'abc':'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
}


class Python310FileDigestCompatTests(unittest.TestCase):
    """Board evidence 2026-09-24: X5/S100 run Python 3.10 and have no file_digest."""
    def test_digest_matches_known_sha256_without_file_digest(self):
        """Digests stay correct on 3.10: empty file, FIPS vectors and block boundaries."""
        chunk=1024*1024
        base=bytes((i*131+7)%256 for i in range(2*chunk+17))
        sizes=[0,1,1023,1024,1025,65535,65536,65537,chunk-1,chunk,chunk+1,2*chunk+17]
        with tempfile.TemporaryDirectory() as td:
            tmp=Path(td)
            with _NoFileDigest():
                self.assertFalse(hasattr(hashlib,'file_digest'))
                for size in sizes:
                    payload=base[:size]
                    path=tmp/f'pattern-{size}.bin';path.write_bytes(payload)
                    expected=hashlib.sha256(payload).hexdigest()
                    with self.subTest(size=size):
                        self.assertEqual(_digest(path if size%2 else str(path)),expected)
                for payload,expected in _KNOWN_SHA256.items():
                    path=tmp/f'known-{len(payload)}.bin';path.write_bytes(payload)
                    with self.subTest(known=len(payload)):
                        self.assertEqual(_digest(path),expected)

    def _run_board_cli(self,sample,target,tmp,sabotage=None,raise_on=None):
        """Drive main() like the board command, on fixture models and fake SDK."""
        ep=tmp/'encoder.fixture';ep.write_bytes(b'host-fixture-encoder')
        dp=tmp/'decoder.fixture';dp.write_bytes(b'host-fixture-decoder')
        base=resolve_selection(sample,target)
        argv=['--target',target,'--output-dir',str(tmp/'evidence'),
              '--encoder-model-path',str(ep),'--decoder-model-path',str(dp),
              '--encoder-asset-id',base.encoder_asset.reference,
              '--decoder-asset-id',base.decoder_asset.reference]
        calls=[]
        def create(path):
            stage='encoder' if Path(path)==ep else 'decoder'
            fake=FakeRuntime(metadata(sample,stage,target))
            calls.append(fake)
            if stage=='decoder':fake.outputs['iou_predictions'].reshape(-1)[:]=[0.1,0.9,0.2]
            if sabotage is not None and len(calls)==4:sabotage(fake)
            if raise_on is not None and len(calls)==2:
                def broken_run(inputs,boom=raise_on):raise boom
                fake.run=broken_run
            return fake
        with patch('samples._shared.sam_evaluator.require_execution_target',return_value=target),\
             patch('samples._shared.model_runner._default_runtime_factory',return_value=create):
            return main(sample,argv)

    def test_board_cli_passes_without_file_digest(self):
        """The exact board failure shape (missing file_digest) must complete with rc 0."""
        for sample,target in (('efficient_sam','s100'),('mobile_sam','x5')):
            with self.subTest(sample=sample,target=target),tempfile.TemporaryDirectory() as td:
                tmp=Path(td)
                with _NoFileDigest():
                    rc=self._run_board_cli(sample,target,tmp)
                self.assertEqual(rc,0)
                stored=json.loads((tmp/'evidence'/'comparison.json').read_text())
                self.assertTrue(stored['passed'],stored.get('error'))
                self.assertEqual(stored['artifacts']['encoder']['observed_sha256'],
                                 hashlib.sha256(b'host-fixture-encoder').hexdigest())
                self.assertEqual(stored['artifacts']['decoder']['observed_sha256'],
                                 hashlib.sha256(b'host-fixture-decoder').hexdigest())
                self.assertGreaterEqual(len(stored['code_sha256']),8)

    def test_board_cli_reports_comparison_failure_as_one(self):
        """Completed runs with a failed check still exit 1 with full evidence."""
        with tempfile.TemporaryDirectory() as td:
            tmp=Path(td)
            def sabotage(fake):
                fake.outputs['iou_predictions'].reshape(-1)[:]=[0.9,0.1,0.2]
                fake.outputs['low_res_masks'][0,0,0,0]=-3
            rc=self._run_board_cli('efficient_sam','s100',tmp,sabotage=sabotage)
            self.assertEqual(rc,1)
            stored=json.loads((tmp/'evidence'/'comparison.json').read_text())
            self.assertFalse(stored['passed']);self.assertEqual(stored['return_code'],1)
            self.assertEqual(len(stored['arrays']),14)
            self.assertFalse(stored['checks']['mask_equal'])
            self.assertGreater(stored['mask_changed_pixels'],0)

    def test_cli_maps_execution_exception_to_error_code_two(self):
        """README contract: execution failure exits 2, not an unhandled traceback."""
        boom=AttributeError("module 'hashlib' has no attribute 'file_digest'")
        with tempfile.TemporaryDirectory() as td:
            argv=['--target','s100','--output-dir',str(Path(td)/'evidence')]
            with patch('samples._shared.sam_evaluator.run_comparison',side_effect=boom),\
                 redirect_stderr(StringIO()) as err:
                rc=main('efficient_sam',argv)
            self.assertEqual(rc,2)
            self.assertIn('error:',err.getvalue())

    def test_execution_exception_after_capture_keeps_evidence(self):
        """An unexpected exception mid-run keeps captured arrays and exits 2."""
        with tempfile.TemporaryDirectory() as td:
            tmp=Path(td)
            rc=self._run_board_cli('efficient_sam','s100',tmp,
                raise_on=AttributeError('simulated SDK failure'))
            self.assertEqual(rc,2)
            stored=json.loads((tmp/'evidence'/'comparison.json').read_text())
            self.assertEqual(stored['error']['type'],'AttributeError')
            self.assertEqual(stored['return_code'],2)
            self.assertFalse(stored['passed'])
            self.assertGreaterEqual(len(stored['arrays']),3)


class _BoardQuantParams:
    """Mimics hbm_runtime.QuantParams: attributes read, any copy refuses (X5 board evidence 2026-09-24)."""
    def __init__(self,quant_type='SCALE',scale=0.25,zero_point=7,axis=3):
        self.quant_type=types.SimpleNamespace(name=quant_type)
        self.scale=np.asarray(scale,dtype=np.float32)
        self.zero_point=np.asarray(zero_point,dtype=np.int32)
        self.axis=axis
    def __deepcopy__(self,memo):raise TypeError("cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object")
    def __copy__(self):raise TypeError("cannot pickle 'hbm_runtime.HB_HBMRuntime.QuantParams' object")


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

    def test_metadata_evidence_survives_copy_hostile_board_quant_params(self):
        """The old asdict() metadata snapshot raised TypeError on the real board."""
        with tempfile.TemporaryDirectory() as td:
            tmp=Path(td)
            image=np.zeros((3,5,3),dtype=np.uint8)
            img=tmp/'input.raw'; img.write_bytes(image.tobytes())
            base=resolve_selection('efficient_sam','s100')
            ep=tmp/'encoder.fixture';ep.write_bytes(b'host-fixture-encoder')
            dp=tmp/'decoder.fixture';dp.write_bytes(b'host-fixture-decoder')
            selection=resolve_selection('efficient_sam','s100',encoder_model_path=ep,decoder_model_path=dp,
                encoder_asset_id=base.encoder_asset.reference,decoder_asset_id=base.decoder_asset.reference)
            def factory(path):
                stage='encoder' if Path(path)==ep else 'decoder'
                fake=FakeRuntime(metadata('efficient_sam',stage,'s100'))
                # SDK-like vestigial descriptor riding along the F32 outputs.
                fake.output_quants={n:_BoardQuantParams() for n in fake.outputs}
                if stage=='decoder': fake.outputs['iou_predictions'].reshape(-1)[:]=[0.1,0.9,0.2]
                return fake
            with patch('samples._shared.sam_evaluator.require_execution_target',return_value='s100'):
                summary=run_comparison(selection,image,img,tmp/'evidence',runtime_factory=factory)
            self.assertTrue(summary['passed'],summary.get('error'))
            # Raw tensors are still captured and the comparison untouched.
            self.assertTrue(summary['checks']['raw_close'])
            self.assertTrue(summary['checks']['inputs_equal'])
            self.assertEqual(summary['mask_changed_pixels'],0)
            # Every side x stage snapshot keeps the full quant descriptor.
            for side in ('legacy','unified'):
                for stage in ('encoder','decoder'):
                    for quant in summary['metadata'][side][stage]['output_quants'].values():
                        self.assertEqual(quant['quant_type'],'SCALE')
                        self.assertEqual(quant['scale'],0.25)
                        self.assertEqual(quant['zero_point'],7)
                        self.assertEqual(quant['axis'],3)
            stored=json.loads((tmp/'evidence/comparison.json').read_text())
            quant=stored['metadata']['legacy']['encoder']['output_quants']['image_embeddings']
            self.assertEqual(quant['scale'],0.25)
            self.assertEqual(quant['zero_point'],7)
            self.assertEqual(quant['axis'],3)
            self.assertTrue((tmp/'evidence/legacy_encoder_outputs_image_embeddings.npy').is_file())

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
