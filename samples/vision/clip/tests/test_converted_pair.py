"""A converted CLIP pair authenticates both files before either runtime loads."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from utils.py_utils.assets import Asset
from samples.vision.clip.runtime.python import cli
from samples.vision.clip.runtime.python.matching import RuntimeModelRunner
from test_clip import ImageRuntime, TextSession


class ConvertedPairTests(unittest.TestCase):
    def test_corrupt_text_rejects_before_image_or_text_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=root/'image.hbm';text=root/'text.onnx'
            image.write_bytes(b'image fixture');text.write_bytes(b'text fixture')
            assets=[Asset('s','clip','s100/'+p.name,fmt,'https://example.invalid/'+p.name,
                          hashlib.sha256(p.read_bytes()).hexdigest()) for p,fmt in ((image,'hbm'),(text,'onnx'))]
            registry={'test-profile':{'s100':{role:{'filename':a.filename,'sha256':a.sha256}
                                             for role,a in zip(('image','text'),assets)}}}
            image_factory=Mock(return_value=ImageRuntime());text_factory=Mock(return_value=TextSession())
            with patch.object(cli,'converted_pairs',return_value=registry),patch.object(cli,'list_assets',return_value=assets):
                selected=cli.resolve_selection('s100',variant='test-profile',image_model_path=image,text_model_path=text,
                    image_asset_id=assets[0].reference,text_asset_id=assets[1].reference)
                text.write_bytes(b'wrong text model')
                runner=RuntimeModelRunner(selected,image_factory=image_factory,text_factory=text_factory)
                with self.assertRaisesRegex(ValueError,'model bytes differ'):
                    runner.load()
                image_factory.assert_not_called();text_factory.assert_not_called()
                self.assertFalse(runner.loaded)
                text.write_bytes(b'text fixture')
                runner.load()
                self.assertTrue(runner.loaded)
                self.assertEqual(image_factory.call_count,1);self.assertEqual(text_factory.call_count,1)

    def test_incomplete_pair_does_not_create_target_support(self):
        with patch.object(cli,'converted_pairs',return_value={'bad':{'s600':{'image':{}}}}):
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                cli.resolve_selection('s600',variant='bad')
