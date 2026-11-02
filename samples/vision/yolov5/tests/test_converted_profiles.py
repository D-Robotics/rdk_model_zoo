"""New raw-F32 profiles do not inherit the historical S dequantization ABI."""
import io
import json
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import numpy as np
from samples.vision.yolov5.runtime.python import cli, detection
from samples.vision.yolov5.runtime.cpp import launcher
from utils.py_utils.assets import Asset
from utils.py_utils.quantization import apply_output_transform
from utils.py_utils.runtime_meta import MetadataMismatchError


class ConvertedProfileTests(unittest.TestCase):
    def setUp(self):
        self.original = cli.list_assets

    def rows(self, group, consumer):
        rows = list(self.original(group, consumer))
        if consumer == 'yolov5':
            for target in (('x5',) if group == 'x5' else ('s100', 's600')):
                for variant in cli.CONVERTED_VARIANTS:
                    filename = cli.converted_filename(target, variant)
                    if filename not in {a.filename for a in rows}:
                        rows.append(Asset(group, consumer, filename, 'bin' if group == 'x5' else 'hbm',
                                          'https://example.test/' + filename, 'a'*64))
        return tuple(rows)

    def metadata(self, target):
        inputs = {'images': (1,3,640,640)} if target == 'x5' else {'images_y': (1,640,640,1), 'images_uv': (1,320,320,2)}
        names = ('small','medium','big')
        return dict(model_name='native', model_names=('native',), input_names=tuple(inputs),
                    input_shapes=inputs, input_dtypes={n:'uint8' for n in inputs}, output_names=names,
                    output_shapes={n:(1,640//s,640//s,255) for n,s in zip(names,(8,16,32))},
                    output_dtypes={n:'float32' for n in names},
                    output_quants={n:types.SimpleNamespace(quant_type='SCALE',scale=np.array([0.01]),zero_point=np.array([17]),axis=3) for n in names})

    def test_defaults_preserved_and_each_converted_identity_is_explicit(self):
        with patch.object(cli,'list_assets',side_effect=self.rows):
            self.assertEqual(cli.resolve_selection('x5').variant,'n-v7.0')
            for t in ('s100','s600'):
                self.assertEqual(cli.resolve_selection(t).variant,'x-672')
            for target in ('x5','s100','s600'):
                for v in cli.CONVERTED_VARIANTS:
                    selected = cli.resolve_selection(target,variant=v)
                    self.assertEqual(selected.asset.filename,cli.converted_filename(target,v))
                    self.assertEqual(cli.resolve_selection(target,asset_id=selected.asset.reference).variant,v)

    def test_float_heads_ignore_quant_descriptors_and_reject_integer_storage(self):
        with patch.object(cli,'list_assets',side_effect=self.rows):
            for target in ('s100','s600'):
                selection=cli.resolve_selection(target,variant=cli.CONVERTED_VARIANTS[0])
                metadata=self.metadata(target)
                binding=detection.bind_model(selection,metadata)
                self.assertEqual((binding.input_size,binding.resize_type,binding.output_transform),(640,0,'raw_f32'))
                raw={n:np.full(shape,2.5,np.float32) for n,shape in metadata['output_shapes'].items()}
                converted=apply_output_transform(binding.output_transform,raw,metadata['output_quants'])
                for name in raw:np.testing.assert_array_equal(converted[name],raw[name])
                metadata['output_dtypes']['small']='int32'
                with self.assertRaises(MetadataMismatchError):detection.bind_model(selection,metadata)

    def test_new_s_profile_reports_640_and_does_not_enter_old_cpp_adapter(self):
        with patch.object(cli,'list_assets',side_effect=self.rows):
            selected=cli.resolve_selection('s100',variant=cli.CONVERTED_VARIANTS[0])
            out=io.StringIO()
            with redirect_stdout(out):cli.dry_run(selected,cli.SAMPLE_DIR/'test_data/bus.jpg')
            self.assertEqual(json.loads(out.getvalue())['input_protocol'],'split_nv12_640')
            args=launcher.build_parser().parse_args(['--target','s100','--variant',cli.CONVERTED_VARIANTS[0]])
            with self.assertRaisesRegex(ValueError,'Python runtime'):launcher._selection(args)


if __name__ == '__main__':
    unittest.main()
