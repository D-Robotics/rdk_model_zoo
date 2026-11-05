"""The rebuilt pair keeps its real 97-class ABI on each independent S target."""
import json
from pathlib import Path
import unittest

from samples.vision.paddle_ocr.runtime.python.cli import BindingError, list_available_pairs, resolve_pair
from samples.vision.paddle_ocr.runtime.python.backend import bind_stage


class ConvertedPairTests(unittest.TestCase):
    def test_new_pairs_bind_collected_hbm_io_and_keep_legacy_defaults(self):
        root=Path(__file__).resolve().parents[1]
        profiles=json.loads((root/'conversion/converted_targets_20261104.json').read_text())['profiles']
        converted=[pair for pair in list_available_pairs()if pair.variant.endswith('-20261104-r02')]
        self.assertEqual({p.target for p in converted},{'s100','s100p','s600'})
        for pair in converted:
            self.assertEqual(len(pair.vocabulary.load_tokens()),97)
            self.assertEqual(pair.vocabulary.load_tokens()[-2:],(' ',' '))
            for role,shape in [('detector',(1,3,640,640)),('recognizer',(1,3,48,320))]:
                contract=getattr(pair,role)
                profile,=[p for p in profiles if p['target']['platform']==pair.target and
                          ('640x640' if role=='detector' else '48x320')in p['variant']]
                io=profile['io'];facts={'model_name':contract.model_name}
                for single,plural in [('input','inputs'),('output','outputs')]:
                    facts.update({single+'_names':[t['name']for t in io[plural]],
                                  single+'_shapes':{t['name']:t['shape']for t in io[plural]},
                                  single+'_dtypes':{t['name']:t['dtype']for t in io[plural]}})
                binding=bind_stage(pair,role,facts)
                self.assertEqual(binding.contract.input_shapes['x'],shape)
                self.assertEqual(binding.contract.output_shape,(1,1,640,640)if role=='detector'else(1,40,97))
        self.assertEqual(resolve_pair('x5').variant,'ppocrv3')
        self.assertEqual(resolve_pair('s100').variant,'ppocrv6')

    def test_cross_target_and_cross_generation_pairs_are_rejected(self):
        pairs={p.target:p for p in list_available_pairs()if p.variant.endswith('-20261104-r02')}
        old=resolve_pair('s100')
        for det,rec in [(pairs['s100'],pairs['s600']),(old,pairs['s100']),(pairs['s100'],old)]:
            with self.assertRaises(BindingError):
                resolve_pair('auto',det_asset_id=det.detector_asset,rec_asset_id=rec.recognizer_asset)


if __name__=='__main__':unittest.main()
