"""Host tests for the H5 platform deployment profiles."""

from __future__ import annotations

import unittest

from samples._shared.platform_profile import (
    ARCHIVE_ROOT,
    UnsupportedProfileError,
    classification_profiles,
    resolve_profile,
)


class ClassificationProfilesTests(unittest.TestCase):
    def _table(self):
        return classification_profiles(url_prefix_s="rdk_s100/MobileNet")

    def test_standard_table_declares_exactly_four_platforms(self):
        self.assertEqual(
            tuple(sorted(self._table())), ("s100", "s100p", "s600", "x5")
        )

    def test_x5_profile_matches_flat_bin_delivery(self):
        profile = self._table()["x5"]
        self.assertEqual(profile.family, "x5")
        self.assertEqual(profile.model_format, ".bin")
        self.assertEqual(profile.model_subdir, "")
        self.assertEqual(profile.march, "bayes-e")
        self.assertFalse(profile.supports_cpp)

    def test_s_profiles_match_hbm_subdirectory_delivery(self):
        for key in ("s100", "s600"):
            profile = self._table()[key]
            self.assertEqual(profile.family, "s")
            self.assertEqual(profile.model_format, ".hbm")
            self.assertEqual(profile.model_subdir, key)
        self.assertEqual(self._table()["s100"].march, "nash-e")
        self.assertEqual(self._table()["s600"].march, "nash-p")

    def test_s_url_prefix_flows_into_artifact_urls(self):
        table = self._table()
        self.assertEqual(
            table["s100"].model_base_url(),
            f"{ARCHIVE_ROOT}/rdk_s100/MobileNet",
        )
        # The S600 artifacts live under their own archive directory; the
        # sample passes the s100 prefix and the profile owns the mapping.
        self.assertEqual(
            table["s600"].model_base_url(),
            f"{ARCHIVE_ROOT}/rdk_s600/MobileNet",
        )

    def test_s100p_base_url_raises_instead_of_guessing(self):
        with self.assertRaises(UnsupportedProfileError):
            self._table()["s100p"].model_base_url()

    def test_s100p_publishes_nothing_and_declares_that(self):
        profile = self._table()["s100p"]
        self.assertIsNone(profile.url_prefix)
        self.assertEqual(profile.model_format, ".hbm")

    def test_supports_cpp_flags_are_off_by_default_and_settable(self):
        default = self._table()
        self.assertFalse(default["x5"].supports_cpp)
        self.assertFalse(default["s100"].supports_cpp)
        flagged = classification_profiles(
            url_prefix_s="rdk_s100/MobileNet", supports_cpp_s=True
        )
        self.assertTrue(flagged["s100"].supports_cpp)
        self.assertTrue(flagged["s600"].supports_cpp)
        self.assertFalse(flagged["x5"].supports_cpp)


class ResolveProfileTests(unittest.TestCase):
    def _table(self):
        return classification_profiles(url_prefix_s="rdk_s100/MobileNet")

    def test_explicit_platform_name_resolves(self):
        profile = resolve_profile(self._table(), "x5")
        self.assertEqual(profile.key, "x5")

    def test_unknown_platform_is_rejected_not_guessed(self):
        with self.assertRaises(UnsupportedProfileError):
            resolve_profile(self._table(), "x3")

    def test_soc_name_maps_through_declared_socs(self):
        table = self._table()
        profile = resolve_profile(table, soc_name="s600")
        self.assertEqual(profile.key, "s600")

    def test_none_arguments_raise_instead_of_falling_back(self):
        with self.assertRaises(UnsupportedProfileError):
            resolve_profile(self._table(), None)


if __name__ == "__main__":
    unittest.main()
