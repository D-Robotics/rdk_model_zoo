"""Independently converted raw-F32 heads keep their 672 stretch geometry."""
import json
from pathlib import Path
import unittest
from samples.vision.bytetrack.runtime.python.cli import resolve_selection
from samples.vision.yolov5.runtime.python.detection import bind_model
from samples.vision.yolov5.runtime.python.cli import BYTETRACK_CONVERTED_VARIANTS

class ConvertedByteTrackTests(unittest.TestCase):
    def test_all_three_targets_bind_the_independently_collected_raw_head_contracts(self):
        path=Path(__file__).resolve().parents[1]/'conversion/converted_targets_20261104.json'
        profiles=json.loads(path.read_text())['profiles']
        self.assertEqual({p['target']['platform']for p in profiles},{'s100','s100p','s600'})
        self.assertEqual(len(profiles),3)
        for profile in profiles:
            target=profile['target']['platform'];selection=resolve_selection(target,variant=profile['variant'])
            self.assertEqual(selection.asset.sha256,profile['components']['model']['sha256'])
            self.assertTrue(selection.asset.filename.startswith(target+'/'))
            facts={'model_name':'compiled-detector'}
            for singular,plural in [('input','inputs'),('output','outputs')]:
                rows=profile['io'][plural]
                facts.update({singular+'_names':[r['name']for r in rows],
                              singular+'_shapes':{r['name']:r['shape']for r in rows},
                              singular+'_dtypes':{r['name']:r['dtype']for r in rows}})
            binding=bind_model(selection,facts)
            self.assertEqual((binding.input_size,binding.resize_type,binding.output_transform),(672,0,'raw_f32'))
            self.assertEqual(resolve_selection(target,asset_id=selection.asset.reference).variant,profile['variant'])

    def test_new_detector_keeps_raw_outputs_and_rejects_cross_consumer(self):
        variant, = BYTETRACK_CONVERTED_VARIANTS
        selection = resolve_selection('s100', variant=variant)
        facts={'model_name':'detector','input_names':['images_y','images_uv'],
            'input_shapes':{'images_y':[1,672,672,1],'images_uv':[1,336,336,2]},
            'input_dtypes':{'images_y':'uint8','images_uv':'uint8'},
            'output_names':['small','medium','big'],
            'output_shapes':{n:[1,h,h,255]for n,h in [('small',84),('medium',42),('big',21)]},
            'output_dtypes':{n:'float32'for n in ['small','medium','big']}}
        binding=bind_model(selection, facts)
        self.assertEqual((binding.input_size,binding.resize_type,binding.output_transform),(672,0,'raw_f32'))
        self.assertEqual(dict(binding.output_quants),{})
        self.assertEqual(resolve_selection('s100',asset_id=selection.asset.reference).variant,variant)
        with self.assertRaises(ValueError):resolve_selection('s600',asset_id=selection.asset.reference)
        from samples.vision.yolov5.runtime.python.cli import resolve_selection as detector_selection
        with self.assertRaises(ValueError):detector_selection('s100',asset_id=selection.asset.reference)
