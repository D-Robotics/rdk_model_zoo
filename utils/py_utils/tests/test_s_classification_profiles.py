"""Validate newly converted S classifiers using their actual compiler metadata."""
import importlib
import json
from pathlib import Path
import unittest

from utils.py_utils.assets import resolve_asset
from utils.py_utils.runtime_meta import RuntimeMetadata

ROOT=Path(__file__).resolve().parents[3]
FAMILIES=('convnext','edgenext','efficientformer','efficientformerv2','efficientnet','efficientvit',
          'fasternet','fastvit','hgnetv2','mobileone','repghost','repvgg','repvit','resnext','googlenet')


class SClassificationProfiles(unittest.TestCase):
    def test_every_registered_profile_binds_its_actual_hbm_abi(self):
        checked=0
        for family in FAMILIES:
            path=ROOT/'samples/vision'/family/'conversion/converted_targets_20261103.json'
            if not path.exists():continue
            cli=importlib.import_module(f'samples.vision.{family}.runtime.python.cli')
            for profile in json.loads(path.read_text())['profiles']:
                with self.subTest(family=family,variant=profile['variant'],target=profile['target']['platform']):
                    component=profile['components']['model'];target=profile['target']['platform']
                    reference='s:'+family+':'+component['filename']
                    asset=resolve_asset(reference)
                    self.assertEqual(asset.sha256,component['sha256'])
                    selected=cli.resolve_selection(target,variant=profile['variant'],asset_id=reference)
                    self.assertEqual(selected.asset_id,reference)
                    contract=selected.contract
                    self.assertEqual((contract.input_height,contract.input_width,contract.resize_type,contract.resize_shorter),(224,224,2,256))
                    self.assertEqual(contract.input_protocol,'split_nv12')
                    self.assertEqual(contract.output_semantics,'source_declared_logits')
                    self.assertEqual(contract.output_score_policy,'softmax')
                    self.assertEqual(contract.output_transform,'raw_f32')
                    metadata={'model_name':'static_hbm_metadata'}
                    for side in ('input','output'):
                        tensors=component[side+'s']
                        metadata[side+'_names']=[v['name'] for v in tensors]
                        metadata[side+'_shapes']={v['name']:v['shape'] for v in tensors}
                        metadata[side+'_dtypes']={v['name']:v['dtype'] for v in tensors}
                    cli.bind_model(selected,RuntimeMetadata.from_mapping(metadata))
                    for check in ('board_smoke','accuracy','runtime_1','runtime_2','cpp_1','cpp_2'):
                        self.assertEqual(profile['validation'][check],'not-run')
                    self.assertEqual(component['calibration_count'],200)
                    self.assertTrue(profile['source'])
                    checked+=1
        self.assertGreater(checked,0,'No collected S classifier was registered')
