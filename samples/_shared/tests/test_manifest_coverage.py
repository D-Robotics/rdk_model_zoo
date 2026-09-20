"""H6: every unified sample is anchored by exact manifest rows.

The unified release manifests (``docs/release/{x5,s}/models.yaml``) are the
machine-readable support matrix introduced with A4.  A directory under
``samples/`` that carries ``runtime/python/model_binding.py`` declares
itself a migrated, manifest-backed sample; these tests keep that claim and
the manifests in lock-step so neither drifts silently during the batches.
Rows for batches that have not migrated yet legitimately point at paths
that do not exist on disk yet — only migrated samples are checked here.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from samples._shared.assets import _ROOT, _models

BINDING_MARKER = "runtime/python/model_binding.py"


def _unified_samples() -> list[str]:
    return sorted(
        binding.parents[2].relative_to(_ROOT).as_posix()
        for binding in _ROOT.glob(f"samples/*/*/{BINDING_MARKER}")
    )


def _manifest_rows() -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for group in ("x5", "s"):
        if (_ROOT / f"docs/release/{group}/models.yaml").is_file():
            rows.extend((group, model) for model in _models(group))
    return rows


class ManifestCoverageTests(unittest.TestCase):
    def setUp(self):
        self.rows = _manifest_rows()
        if not self.rows:
            self.skipTest("no unified manifests under docs/release/ in this checkout")
        self.samples = _unified_samples()

    def _rows_by_sample_path(self) -> dict[str, list[str]]:
        indexed: dict[str, list[str]] = {}
        for group, model in self.rows:
            indexed.setdefault(model.get("sample_path"), []).append(
                f"{group}:{model['id']}"
            )
        return indexed

    def test_the_marker_glob_still_finds_the_pilots_and_b1(self):
        # Guards the degenerate case where a broken glob makes both coverage
        # assertions below pass vacuously.
        self.assertGreaterEqual(
            len(self.samples),
            7,
            f"expected pilots + B1 samples, marker found only {self.samples}",
        )

    def test_every_unified_sample_has_exact_manifest_rows(self):
        indexed = self._rows_by_sample_path()
        uncovered = [s for s in self.samples if s not in indexed]
        if uncovered:
            tails = {row_path.rsplit("/", 1)[-1] for row_path in indexed}
            hints = sorted(
                f"{s} (near-miss ids: {sorted(t for t in tails if t in s) or 'none'})"
                for s in uncovered
            )
            self.fail("unified samples without an exact sample_path row: " + "; ".join(hints))

    def test_download_scripts_of_unified_samples_exist_on_disk(self):
        migrated = set(self.samples)
        missing = []
        for group, model in self.rows:
            if model.get("sample_path") not in migrated:
                continue
            for script in model.get("download_scripts") or []:
                if not (_ROOT / script).is_file():
                    missing.append(f"{group}:{model['id']} -> {script}")
        self.assertEqual(
            missing, [], "manifest lists download scripts that are not on disk"
        )

    def test_unified_sample_rows_never_point_back_at_platforms(self):
        # A4 moved the manifests to docs/release/; a row that regains a
        # platforms/ path would break resolve_asset's sample contract.
        offenders = [
            f"{group}:{model['id']} -> {model.get('sample_path')}"
            for group, model in self.rows
            if str(model.get("sample_path", "")).startswith("platforms/")
        ]
        self.assertEqual(offenders, [], "manifest rows still point under platforms/")


if __name__ == "__main__":
    unittest.main()
