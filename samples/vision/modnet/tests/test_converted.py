"""Converted publication selection keeps the original manual model intact."""
from dataclasses import replace
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from samples.vision.modnet.runtime.python import cli
from samples.vision.modnet.model import download
from utils.py_utils.assets import Asset, list_assets


class ConvertedModelTests(unittest.TestCase):
    def manifest(self, rows):
        return lambda group, sample: rows if group == "x5" else list_assets(group, sample)

    def setUp(self):
        self.legacy = Asset("x5", "modnet", "modnet_512x512_rgb.bin", "bin", None, None)
        self.converted = Asset("x5", "modnet", cli.CONVERTED_ASSET_ID.split(":", 2)[2],
                               "bin", "https://example.test/modnet.bin", "a" * 64)

    def test_new_publication_does_not_change_default(self):
        with patch.object(cli, "list_assets", side_effect=self.manifest((self.legacy, self.converted))):
            self.assertEqual(cli.resolve_selection("x5").asset, self.legacy)
            selected = cli.resolve_selection("x5", asset_id=cli.CONVERTED_ASSET_ID)
            self.assertEqual(selected.asset, self.converted)
            self.assertEqual(selected.model_path.name, self.converted.filename)
            self.assertEqual(cli.list_available_assets("x5"), (self.legacy, self.converted))

    def test_unregistered_or_unknown_graph_cannot_take_the_native_binding(self):
        with patch.object(cli, "list_assets", side_effect=self.manifest((self.legacy,))):
            with self.assertRaises(cli.BindingError):
                cli.resolve_selection("x5", asset_id=cli.CONVERTED_ASSET_ID)
        unknown = replace(self.converted, filename="different_shape.bin")
        with patch.object(cli, "list_assets", side_effect=self.manifest((self.legacy, unknown))):
            with self.assertRaises(cli.BindingError):
                cli.resolve_selection("x5")

    def test_explicit_converted_download_uses_manifest_integrity_path(self):
        with patch.object(cli, "list_assets", side_effect=self.manifest((self.legacy, self.converted))), \
                patch("utils.py_utils.assets.download_asset", return_value="a" * 64) as fetch:
            self.assertEqual(download.main(["--target", "x5", "--asset-id", cli.CONVERTED_ASSET_ID]), 0)
            self.assertEqual(fetch.call_args.args[0], self.converted)
            self.assertEqual(fetch.call_args.args[1].name, self.converted.filename)
            fetch.reset_mock()
            self.assertEqual(download.main(["--target", "x5"]), 2)
            fetch.assert_not_called()

    def test_registered_s_targets_are_listed_with_their_actual_identity(self):
        registry = json.loads((cli.SAMPLE_DIR / "model/converted_assets.json").read_text())
        for targets in registry["profiles"].values():
            for target, expected in targets.items():
                with self.subTest(target=target):
                    selection = cli.resolve_selection(target, asset_id="s:modnet:" + expected["filename"])
                    self.assertEqual(selection.asset.sha256, expected["sha256"])
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        self.assertEqual(cli.run_list_models(target), 0)
                    rows = json.loads(output.getvalue())
                    self.assertTrue(rows)
                    self.assertTrue(all(row["target"] == target for row in rows))


if __name__ == "__main__":
    unittest.main()
