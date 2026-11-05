"""An exact new asset must use its own file instead of the historical default."""
import unittest
from unittest.mock import patch

from utils.py_utils.assets import Asset
from samples.vision.yolo26_depth.runtime.python import cli


class ConvertedAssetTests(unittest.TestCase):
    def test_new_asset_id_selects_its_own_default_path(self):
        asset = Asset('s','yolo26_depth','nash-e/yolo26n_depth_new_int16.hbm','hbm',
                      'https://example.org/yolo26n_depth_new_int16.hbm','a'*64)
        with patch.object(cli,'list_available_assets',return_value=(asset,)):
            selected = cli.resolve_selection('s100',asset_id=asset.reference)
        self.assertEqual(selected.model_path,cli.SAMPLE_DIR/'model'/asset.filename)
        self.assertEqual(selected.asset,asset)
        self.assertEqual(selected.variant,'n')


if __name__ == '__main__':
    unittest.main()
