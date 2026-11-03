"""Single-platform converted pairs retain their exact download/runtime identity."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from samples.vision.fasterrcnn.model import download
from samples.vision.fasterrcnn.runtime.python import cli


class ConvertedVariantTests(unittest.TestCase):
    def test_download_defaults_to_the_selected_variants_actual_targets(self):
        pairs = {'s100':object()}
        with patch.object(download,'descriptor_pairs',return_value=pairs),patch.object(download,'download_target') as get:
            self.assertEqual(download.main(['--variant','test-converted']),0)
            self.assertEqual([call.args[0] for call in get.call_args_list],['s100'])

    def test_missing_explicit_target_is_a_reported_user_error(self):
        with patch.object(download,'descriptor_pairs',return_value={'s100':object()}),patch.object(download,'download_target') as get:
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(download.main(['--variant','test-converted','--target','x5']),2)
            self.assertIn('no published x5 pair',errors.getvalue())
            get.assert_not_called()

    def test_named_converted_pair_keeps_both_hashes_with_external_paths(self):
        pair = cli.TargetPair('s100','nash-e','parta.hbm','a'*64,'partb.hbm','b'*64,'packed',{})
        with patch.object(cli,'descriptor_pairs',return_value={'s100':pair}):
            selected = cli.resolve_selection('s100',variant='test-converted',
                                              parta_path=Path('/tmp/parta.hbm'),partb_path=Path('/tmp/partb.hbm'))
        self.assertEqual((selected.parta_sha256,selected.partb_sha256),('a'*64,'b'*64))
        self.assertEqual(selected.asset_id,pair.asset_id)


if __name__ == '__main__':
    unittest.main()
