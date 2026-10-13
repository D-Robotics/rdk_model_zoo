"""Host background preparation: the sidecar is measured, not assumed.

Runs the real script against a real FFmpeg on tiny generated fixtures
(skipped where ffmpeg/ffprobe are unavailable).  The source mirrors the
bundled asset: an MPEG-4 Part 2 mp4 at a width that is not an 8-pixel
multiple, so the pad path and the visible/coded width split are both
exercised, and every sidecar field is checked against the produced files.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from samples.vision.lite_avatar.runtime.python import media_worker

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "test_data" / "prepare_background.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_background", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "FFmpeg not on PATH",
)
class PrepareBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # 30 visible columns (coded 32 exercises the pad), MPEG-4 Part 2
        # like the bundled asset, 15 frames at 30 fps.
        source = self.root / "bg_video.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=30x48:rate=30",
                "-t", "0.5", "-c:v", "mpeg4", str(source),
            ],
            check=True,
        )
        self.source_bytes = source.read_bytes()
        self.prepare = _load_script()

    def test_sidecar_is_measured_from_the_produced_stream(self):
        with contextlib.redirect_stdout(io.StringIO()):
            sidecar = self.prepare.prepare(self.root / "bg_video.mp4", self.root)
        self.assertEqual(sidecar["codec"], "h264")
        self.assertEqual(sidecar["width"], 30)  # visible, from the source
        self.assertEqual(sidecar["coded_width"], 32)  # 8-pixel multiple
        self.assertEqual(sidecar["height"], 48)
        self.assertEqual(sidecar["fps"], 30)
        self.assertEqual(sidecar["frames"], 15)
        produced = self.root / "bg_video.h264"
        self.assertEqual(
            sidecar["sha256"], hashlib.sha256(produced.read_bytes()).hexdigest()
        )
        on_disk = json.loads(
            (self.root / "bg_video.json").read_text(encoding="utf-8")
        )
        self.assertEqual(on_disk, sidecar)
        # The runtime's own validation accepts what the script wrote.
        spec = media_worker.read_background_sidecar(self.root)
        self.assertEqual(
            (spec["width"], spec["coded_width"], spec["height"], spec["frames"]),
            (30, 32, 48, 15),
        )

    def test_the_original_mp4_is_never_modified(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.prepare.prepare(self.root / "bg_video.mp4", self.root)
        self.assertEqual(
            (self.root / "bg_video.mp4").read_bytes(), self.source_bytes
        )

    def test_missing_source_fails_loudly(self):
        with self.assertRaises(SystemExit):
            self.prepare.prepare(self.root / "absent.mp4", self.root)

    def test_width_is_padded_to_the_decoder_stride_alignment(self):
        # 24 visible columns are an 8-pixel multiple but not 16: coding at
        # 24 would hand the decoder a tight buffer (truncated UV); the
        # preparation must code at the next 16-pixel multiple.
        source = self.root / "narrow.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=24x48:rate=30",
                "-t", "0.5", "-c:v", "mpeg4", str(source),
            ],
            check=True,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            sidecar = self.prepare.prepare(source, self.root)
        self.assertEqual((sidecar["width"], sidecar["coded_width"]), (24, 32))
        spec = media_worker.read_background_sidecar(self.root)
        self.assertEqual((spec["width"], spec["coded_width"]), (24, 32))


if __name__ == "__main__":
    unittest.main()
