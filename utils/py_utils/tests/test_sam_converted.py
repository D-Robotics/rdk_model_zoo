"""Converted SAM pairs require both verified files before loading any SDK model."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from utils.py_utils.assets import Asset
from utils.py_utils import sam_binding
from utils.py_utils.sam_runner import RuntimeModelRunner
from utils.py_utils.tests.test_sam_binding import metadata


class ConvertedSAMTests(unittest.TestCase):
    def test_pair_identity_and_float_boundary_are_preserved(self):
        assets = tuple(Asset('s','efficient_sam',f'nash-e/new_{r}.hbm','hbm',
                             f'https://example.org/new_{r}.hbm',d*64)
                       for r,d in [('encoder','a'),('decoder','b')])
        registry = {'converted-test':{'s100':{r:{'filename':a.filename,'sha256':a.sha256}
                                              for r,a in zip(('encoder','decoder'),assets)}}}
        with patch.object(sam_binding,'converted_pairs',return_value=registry),patch.object(sam_binding,'list_assets',return_value=assets):
            selected = sam_binding.resolve_selection('efficient_sam','s100',variant='converted-test')
            self.assertEqual(selected.variant,'converted-test')
            self.assertEqual(selected.encoder_asset.sha256,'a'*64)
            sam_binding.bind_model(selected,metadata('efficient_sam','encoder'),metadata('efficient_sam','decoder'))
            with self.assertRaises(ValueError):
                sam_binding.bind_model(selected,metadata('efficient_sam','encoder',dtype='int16'),metadata('efficient_sam','decoder'))

    def test_wrong_decoder_bytes_are_rejected_before_either_sdk_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = []
            registry = {'converted-test':{'s100':{}}}
            for role in ('encoder','decoder'):
                filename = f'nash-e/new_{role}.hbm'
                data = role.encode()
                path = root/'model'/filename
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                assets.append(Asset('s','efficient_sam',filename,'hbm','https://example.org/'+role,digest))
                registry['converted-test']['s100'][role] = {'filename':filename,'sha256':digest}
            with patch.object(sam_binding,'converted_pairs',return_value=registry),patch.object(sam_binding,'list_assets',return_value=assets):
                selected = sam_binding.resolve_selection('efficient_sam','s100',variant='converted-test',sample_dir=root)
                selected.decoder_model_path.write_bytes(b'replaced')
                factory = Mock()
                with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):
                    RuntimeModelRunner(selected,runtime_factory=factory).load()
                factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
