"""Golden-package verifier manifest-completeness checks (SDK-free).

A partial manifest (one to three of the four stages) or rows beyond the four
must be rejected as a manifest defect, never verified as a subset: every
file that IS published may be present with a matching digest, and the run
must still exit 2 naming the gap.
"""

from contextlib import redirect_stdout
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from samples.vision.lite_avatar.model import verify
from samples.vision.lite_avatar.runtime.python import cli


def fake_assets(target, filenames):
    rows = []
    for stage in cli.STAGES:
        filename = cli.FILENAMES[target][stage]
        if filename in filenames:
            rows.append(SimpleNamespace(filename=filename, sha256="0" * 64))
    return rows


class VerifyManifestCompleteness(unittest.TestCase):
    def run_verify(self, target, rows):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            for row in rows:
                path = model_dir / row.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = f"payload {row.filename}".encode()
                path.write_bytes(payload)
                row.sha256 = hashlib.sha256(payload).hexdigest()
            with mock.patch.object(verify, "list_assets", return_value=rows), \
                    redirect_stdout(io.StringIO()) as captured:
                rc = verify.main(["--target", target, "--model-dir", str(model_dir)])
        return rc, captured.getvalue()

    def test_partial_manifest_is_rejected_not_verified(self):
        # Two of four s100 stages published, both files present with exactly
        # matching digests: still a manifest defect, exit 2 naming the gap.
        rows = fake_assets(
            "s100",
            {cli.FILENAMES["s100"]["audio_encoder"], cli.FILENAMES["s100"]["audio2mouth"]},
        )
        rc, output = self.run_verify("s100", rows)
        self.assertEqual(rc, 2)
        self.assertIn("stage not published for s100", output)
        self.assertIn(cli.FILENAMES["s100"]["facegen_encoder"], output)
        self.assertIn(cli.FILENAMES["s100"]["facegen_decoder"], output)
        self.assertIn("exactly the four s100 stages", output)

    def test_extra_manifest_row_is_rejected(self):
        # The four s600 stages plus one surplus row under the same prefix.
        rows = fake_assets("s600", set(cli.FILENAMES["s600"].values()))
        surplus = "s600/lite_avatar_surplus_stage_s600.hbm"
        rows.append(SimpleNamespace(filename=surplus, sha256="0" * 64))
        rc, output = self.run_verify("s600", rows)
        self.assertEqual(rc, 2)
        self.assertIn(surplus, output)
        self.assertIn("beyond the four s600 stages", output)


if __name__ == "__main__":
    unittest.main()
