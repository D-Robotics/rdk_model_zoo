"""Prepared/legacy manifest contracts for evaluation, not dataset acceptance."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np


class EvaluationInputs(unittest.TestCase):
    def test_nonfinite_trailing_and_missing_features_are_not_silently_skipped(self):
        from samples.speech.paraformer.evaluator.inputs import (
            read_manifest,
            load_feature,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "utt_id": "a",
                            "text": "",
                            "feat_length": 1,
                            "feature_file": "a.npy",
                        }
                    ]
                )
            )
            entries, _ = read_manifest(manifest)
            with self.assertRaises(FileNotFoundError):
                load_feature(entries[0])
            path = root / "a.npy"
            values = np.zeros((1, 400, 560), np.float32)
            values[0, 0, 0] = np.nan
            np.save(path, values)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_feature(entries[0])
            values[0, 0, 0] = 0
            np.save(path, values)
            with path.open("ab") as stream:
                stream.write(b"not part of the array")
            with self.assertRaisesRegex(ValueError, "trailing"):
                load_feature(entries[0])

    def test_inconsistent_frame_metadata_and_invalid_digest_are_rejected(self):
        from samples.speech.paraformer.evaluator.inputs import read_manifest

        base = {"utt_id": "a", "text": "", "feat_length": 400}
        for extra in (
            {"original_frames": 399},
            {"original_frames": 401, "truncated": False},
            {"truncated": 1},
            {"feature_sha256": None},
            {"feature_file": ""},
        ):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temp:
                manifest = Path(temp) / "manifest.json"
                manifest.write_text(json.dumps([{**base, **extra}]))
                with self.assertRaises(ValueError):
                    read_manifest(manifest)

    def test_legacy_and_prepared_paths_preserve_same_owned_features(self):
        from samples.speech.paraformer.evaluator.inputs import (
            read_manifest,
            load_feature,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "feats").mkdir()
            path = root / "feats/a.npy"
            np.save(path, np.ones((1, 400, 560), np.float32))
            record = {"utt_id": "a", "text": "你好", "feat_length": 20}
            for extra in (
                {},
                {
                    "feature_file": "feats/a.npy",
                    "feature_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
            ):
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps([{**record, **extra}]))
                entries, digest = read_manifest(manifest, 0)
                array, sha = load_feature(entries[0])
                self.assertEqual(array.shape, (1, 400, 560))
                self.assertTrue(array.flags.owndata)
                self.assertEqual(len(sha), 64)
                self.assertEqual(len(digest), 64)
                self.assertEqual(entries[0].source, {**record, **extra})

    def test_invalid_full_manifest_rejected_even_outside_selected_prefix(self):
        from samples.speech.paraformer.evaluator.inputs import read_manifest

        good = {"utt_id": "a", "text": "", "feat_length": 10}
        for invalid in (
            {**good},
            {**good, "utt_id": "b", "feat_length": None},
            {**good, "utt_id": "b", "feat_length": True},
            {**good, "utt_id": "b", "text": None},
        ):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "manifest.json"
                path.write_text(json.dumps([good, invalid]))
                with self.assertRaises(ValueError):
                    read_manifest(path, 1)

    def test_corrupt_digest_and_wrong_shape_fail_without_type_coercion(self):
        from samples.speech.paraformer.evaluator.inputs import (
            read_manifest,
            load_feature,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "feats").mkdir()
            path = root / "feats/a.npy"
            manifest = root / "manifest.json"
            record = {
                "utt_id": "a",
                "text": "x",
                "feat_length": 1,
                "feature_sha256": "0" * 64,
            }
            np.save(path, np.zeros((1, 400, 560), np.float32))
            manifest.write_text(json.dumps([record]))
            entries, _ = read_manifest(manifest, 0)
            with self.assertRaisesRegex(ValueError, "digest"):
                load_feature(entries[0])
            del record["feature_sha256"]
            manifest.write_text(json.dumps([record]))
            entries, _ = read_manifest(manifest, 0)
            for values in (
                np.zeros((1, 2, 560), np.float32),
                np.zeros((1, 400, 560), np.float64),
            ):
                np.save(path, values)
                with self.assertRaises(ValueError):
                    load_feature(entries[0])


if __name__ == "__main__":
    unittest.main()
