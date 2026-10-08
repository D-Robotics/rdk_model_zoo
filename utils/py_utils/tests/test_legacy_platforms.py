"""Host tests for the pinned historical-platforms reader.

These pin the helper's own contract: non-ASCII tree names materialize
with their real filenames (Git's default ls-tree quoting must not leak
through), missing objects raise with the exact fetch guidance instead of
silently skipping verification, and materialized names map back to their
``platforms/`` tree names. Requires a clone holding the pinned objects
(the default full clone does).
"""

from __future__ import annotations

import unittest

from utils.py_utils.legacy_platforms import (
    PIN,
    legacy_path,
    legacy_tree,
    pinned_name,
)

#: A pinned path whose name Git would C-quote in default ls-tree output.
NON_ASCII = "x3/demos/classification/模型量化部署.md"


class LegacyPlatformsTests(unittest.TestCase):
    def test_non_ascii_path_materializes_with_its_real_name(self):
        materialized = legacy_path(NON_ASCII)

        self.assertTrue(materialized.is_file())
        self.assertEqual(materialized.name, "模型量化部署.md")
        self.assertGreater(materialized.stat().st_size, 0)

    def test_non_ascii_tree_listing_keeps_real_filenames(self):
        tree = legacy_tree("x3/demos/classification")

        names = {p.name for p in tree.iterdir()}
        self.assertIn("模型量化部署.md", names)

    def test_missing_object_raises_with_fetch_guidance(self):
        with self.assertRaises(FileNotFoundError) as raised:
            legacy_path("x5/samples/vision/does_not_exist_anywhere.py")

        message = str(raised.exception)
        self.assertIn("does_not_exist_anywhere.py", message)
        self.assertIn(PIN, message)
        self.assertIn(f"git fetch origin {PIN}", message)

    def test_missing_tree_raises_instead_of_returning_empty(self):
        with self.assertRaises(FileNotFoundError) as raised:
            legacy_tree("x5/samples/vision/never_existed/")

        self.assertIn("never_existed", str(raised.exception))

    def test_pinned_name_round_trips_a_materialized_path(self):
        materialized = legacy_path(NON_ASCII)

        self.assertEqual(pinned_name(materialized), f"platforms/{NON_ASCII}")


if __name__ == "__main__":
    unittest.main()
