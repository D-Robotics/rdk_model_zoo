"""Video writer protocols under injected failures (no board, no real codec).

The hardware writer drives its real ACK/ring protocol against a fake worker
script through the ``worker_command``/``ring_dir`` seams (board defaults are
untouched); the factory rejects every non-hardware kind — there is no
software video encoder in the runtime to fall back to.  Every case checks
both the raised error and the cleanup state.
"""

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from samples.vision.lite_avatar.runtime.python import cli


def make_script(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


GOOD_WORKER = """#!/bin/sh
echo READY
while read line; do
  case "$line" in
    END) echo DONE; exit 0 ;;
    *) echo "ACK $line" ;;
  esac
done
"""

NO_ACK_WORKER = "#!/bin/sh\necho READY\nsleep 30\n"
DYING_WORKER = "#!/bin/sh\nexit 7\n"
READY_THEN_DIE_WORKER = "#!/bin/sh\necho READY\nexit 7\n"
BAD_EXIT_WORKER = """#!/bin/sh
echo READY
while read line; do
  case "$line" in
    END) exit 3 ;;
    *) echo "ACK $line" ;;
  esac
done
"""
ANSI_WORKER = """#!/bin/sh
printf '\\033[0m native codec noise\\n'
echo READY
while read line; do
  case "$line" in
    END) echo DONE; exit 0 ;;
    *) echo "ACK $line" ;;
  esac
done
"""


class WriterFactoryTests(unittest.TestCase):
    def test_unknown_kinds_are_rejected_never_fallen_back_from(self):
        for kind in ("software", "opencv", "ffmpeg", ""):
            with self.subTest(kind=kind), self.assertRaises(ValueError) as raised:
                cli.make_writer(kind, Path("v.mp4"), Path("a.wav"), 16, 12)
            self.assertIn("Unknown video writer kind", str(raised.exception))

    def test_hardware_writer_requires_ffmpeg_on_the_path(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(RuntimeError) as raised:
                cli.make_writer("hardware", Path("v.mp4"), Path("a.wav"), 16, 12)
            self.assertIn("FFmpeg", str(raised.exception))


class HardwareWriterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ring_dir = self.root / "ring"
        self.ring_dir.mkdir()
        self.out = self.root / "video.mp4"
        self.audio = self.root / "a.wav"
        self.audio.write_bytes(b"")
        self.worker = make_script(self.root / "worker.sh", GOOD_WORKER)
        self.kwargs = dict(
            worker_command=[str(self.worker)],
            ring_dir=str(self.ring_dir),
        )

    def _ring_files(self):
        return sorted(p.name for p in self.ring_dir.iterdir())

    def test_handshake_submit_slot_reuse_and_close(self):
        writer = cli.HardwareVideoWriter(self.out, self.audio, 16, 12, **self.kwargs)
        self.assertTrue(writer.started)
        ring_files = self._ring_files()
        self.assertEqual(len(ring_files), 1)
        self.assertTrue(ring_files[0].startswith("lite-avatar-"))
        self.assertTrue(ring_files[0].endswith(".ring"))
        frame = np.zeros((12, 16, 3), np.uint8)
        for value in range(2 * writer.SLOTS):  # forces ACK-driven slot reuse
            frame[:] = value
            writer.submit(frame)
        writer.close()
        self.assertTrue(writer.done)
        self.assertTrue(writer.log.closed)
        self.assertEqual(self._ring_files(), [])  # ring file removed
        self.assertIsNone(writer.ring)

    def test_render_writes_directly_into_a_ring_slot(self):
        writer = cli.HardwareVideoWriter(self.out, self.audio, 16, 12, **self.kwargs)
        try:
            seen = []

            class FakeResult:
                def render_frame(self, index, out=None):
                    seen.append(out)
                    out[:] = index + 1

            writer.render(FakeResult(), 3)
            # The renderer painted straight into ring memory, not a copy.
            self.assertTrue(
                any(
                    np.shares_memory(seen[0], writer.ring[slot])
                    for slot in range(writer.SLOTS)
                )
            )
        finally:
            writer.close()

    def test_missing_ack_times_out_and_cleans_the_ring(self):
        make_script(self.worker, NO_ACK_WORKER)
        writer = cli.HardwareVideoWriter(self.out, self.audio, 16, 12, **self.kwargs)
        writer.ACK_TIMEOUT = 0.5
        with self.assertRaises(TimeoutError):
            for index in range(2 * writer.SLOTS):
                writer.submit(np.zeros((12, 16, 3), np.uint8))
        writer.abort()
        self.assertEqual(self._ring_files(), [])
        self.assertTrue(writer.log.closed)

    def test_worker_death_before_ready_is_an_init_failure(self):
        # No READY line: stdout EOF still releases the init wait, and the
        # constructor aborts and raises without touching the 15 s timeout.
        make_script(self.worker, DYING_WORKER)
        with self.assertRaises(RuntimeError):
            cli.HardwareVideoWriter(self.out, self.audio, 16, 12, **self.kwargs)
        self.assertEqual(self._ring_files(), [])

    def test_worker_crash_right_after_ready_never_silently_succeeds(self):
        # READY followed by immediate death is racy at every step: the exit
        # may surface in the constructor, in a submit, or no earlier than
        # close() -- but it must surface somewhere and clean the ring.
        make_script(self.worker, READY_THEN_DIE_WORKER)
        writer = None
        raised = False
        try:
            writer = cli.HardwareVideoWriter(self.out, self.audio, 16, 12, **self.kwargs)
            for index in range(4):
                writer.submit(np.zeros((12, 16, 3), np.uint8))
            writer.close()
        except RuntimeError:
            raised = True
        finally:
            if writer is not None:
                writer.abort()
        self.assertTrue(raised)
        self.assertEqual(self._ring_files(), [])

    def test_worker_bad_exit_on_end_raises_after_cleanup(self):
        make_script(self.worker, BAD_EXIT_WORKER)
        writer = cli.HardwareVideoWriter(self.out, self.audio, 16, 12, **self.kwargs)
        writer.submit(np.zeros((12, 16, 3), np.uint8))
        with self.assertRaises(RuntimeError) as raised:
            writer.close()
        self.assertIn("exit 3", str(raised.exception))
        self.assertEqual(self._ring_files(), [])
        writer.abort()  # idempotent after cleanup: no error, no stray kill
        self.assertEqual(writer.proc.returncode, 3)

    def test_abort_removes_the_partial_elementary_stream(self):
        # An aborted run must not leave the worker's scratch behind: the
        # ring and the partial .h264 are gone, the encoder log stays for
        # diagnostics.
        writer = cli.HardwareVideoWriter(self.out, self.audio, 16, 12, **self.kwargs)
        writer.submit(np.zeros((12, 16, 3), np.uint8))
        partial = Path(f"{self.out}.h264")
        partial.write_bytes(b"partial elementary stream")
        writer.abort()
        self.assertEqual(self._ring_files(), [])
        self.assertFalse(partial.exists())
        self.assertTrue(Path(f"{self.out}.encoder.log").exists())

    def test_ansi_noise_lines_are_stripped_and_logged(self):
        make_script(self.worker, ANSI_WORKER)
        writer = cli.HardwareVideoWriter(self.out, self.audio, 16, 12, **self.kwargs)
        writer.submit(np.zeros((12, 16, 3), np.uint8))
        writer.close()
        log = Path(f"{self.out}.encoder.log").read_text(encoding="utf-8")
        self.assertIn("native codec noise", log)
        self.assertNotIn("\x1b", log)
        self.assertTrue(writer.done)


if __name__ == "__main__":
    unittest.main()
