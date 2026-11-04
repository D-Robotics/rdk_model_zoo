"""Exercise the accepted S ABI, normalized input and CPU argmax boundary."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from utils.py_utils.assets import resolve_asset
from samples.vision.pp_liteseg.runtime.python.cli import converted_profiles,resolve_selection
from samples.vision.pp_liteseg.runtime.python.pp_liteseg import bind_model,create_runner,PPLiteSegSegmenter

ROOT=Path(__file__).resolve().parents[4]


def metadata(component):
    fields={'model_name':'pp','model_names':['pp']}
    for side in ['input','output']:
        fields[side+'_names']=[v['name']for v in component[side+'s']]
        fields[side+'_shapes']={v['name']:v['shape']for v in component[side+'s']}
        fields[side+'_dtypes']={v['name']:v['dtype']for v in component[side+'s']}
    return fields


class SLogitsTests(unittest.TestCase):
    def profiles(self):
        rows=converted_profiles();self.assertTrue(rows,'No accepted S logits profile')
        return rows

    def test_all_registered_targets_bind_actual_compiler_metadata(self):
        for row in self.profiles():
            with self.subTest(target=row['target']['platform']):
                component=row['components']['model'];reference='s:pp_liteseg:'+component['filename']
                selection=resolve_selection(row['target']['platform'],variant=row['variant'],asset_id=reference)
                self.assertEqual(resolve_asset(reference).sha256,component['sha256'])
                binding=bind_model(selection,metadata(component));self.assertTrue(binding.selection.float_logits)
                gate=component['cpu_reference_float_gate']
                self.assertEqual(gate['status'],'passed');self.assertEqual(gate['onnx']['sha256'],component['onnx_sha256'])
                self.assertEqual(len(gate['cases']),4)
                for check in ['board_smoke','accuracy','runtime_1','runtime_2','cpp_1','cpp_2']:
                    self.assertEqual(row['validation'][check],'not-run')
                wrong=metadata(component);wrong['input_dtypes']={binding.input_name:'NV12'}
                with self.assertRaises(ValueError):bind_model(selection,wrong)

    def test_runtime_normalization_argmax_and_calibration_match(self):
        row=self.profiles()[0];component=row['components']['model'];fields=metadata(component)
        selection=resolve_selection(row['target']['platform'],variant=row['variant'])
        output=component['outputs'][0];shape=tuple(output['shape']);raw=np.zeros(shape,np.float32)
        nchw=shape[1]==19
        if nchw:
            raw[:,13,:,:]=1;raw[:,7,:,:100]=2
        else:
            raw[:,:,:,13]=1;raw[:,:,:100,7]=2
        sdk=SimpleNamespace(**fields,version='host-fixture')
        sdk.run=lambda tensors:{'pp':{output['name']:raw}}
        task=PPLiteSegSegmenter(selection,runner=create_runner(selection,runtime=sdk))
        image=np.empty((3,5,3),np.uint8);image[:]=[0,127,255]
        prepared=task.preprocess(image);value=next(iter(prepared.tensors.values()))
        self.assertEqual(value.shape,(1,3,512,1024));self.assertEqual(value.dtype,np.float32)
        np.testing.assert_allclose(value[0,:,0,0],[1.,127/127.5-1.,-1.],atol=1e-7)
        labels=task.predict(image);self.assertEqual(labels.dtype,np.int32)
        self.assertTrue(np.all(labels[:,:100]==7));self.assertTrue(np.all(labels[:,100:]==13))
        raw.flat[0]=np.nan
        with self.assertRaises(ValueError):task.postprocess(raw)
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);images=root/'images';images.mkdir();cv2.imwrite(str(images/'colors.png'),image)
            result=subprocess.run([sys.executable,str(ROOT/'samples/vision/pp_liteseg/conversion/prepare_calibration.py'),
                '--src',str(images),'--out',str(root/'cal'),'--source-logits'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            np.testing.assert_array_equal(np.load(next((root/'cal').glob('*.npy')),allow_pickle=False),value)
            receipt=json.loads((root/'cal.manifest.json').read_text());self.assertEqual(receipt['count'],1)


if __name__=='__main__':unittest.main()
