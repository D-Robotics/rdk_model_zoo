"""Video writer protocols under injected failures (no board, no real codec).

The software writer runs against a fake ``ffmpeg`` shell script on PATH; the
hardware writer drives its real ACK/ring protocol against a fake worker
script through the ``worker_command``/``ring_dir`` seams (board defaults are
untouched).  Every case checks both the raised error and the cleanup state.
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


GOOD_FFMPEG = "#!/bin/sh\ncat > /dev/null\nexit 0\n"
EXIT_FFMPEG = "#!/bin/sh\nsleep 0.2\nexit 3\n"
# A plain (non-``exec``) child is the real leak shape: SIGKILL of the shell
# alone leaves ``sleep`` holding the pipe read-end open, so the writer must
# kill the whole process group for its thread to drain.
STALL_FFMPEG = "#!/bin/sh\nsleep 30\n"

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


class SoftwareWriterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        make_script(self.root / "bin" / "ffmpeg", GOOD_FFMPEG)
        self.audio = self.root / "a.wav"
        self.audio.write_bytes(b"")
        patcher = mock.patch.dict(
            os.environ, {"PATH": f"{self.root / 'bin'}:{os.environ['PATH']}"}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.out = self.root / "video.mp4"

    def test_happy_path_submits_and_closes(self):
        writer = cli.SoftwareVideoWriter(self.out, self.audio, 16, 12)
        for value in range(3):
            writer.submit(f"frame{value}".encode())
        writer.close()
        self.assertTrue(writer.log.closed)
        self.assertEqual(writer.proc.returncode, 0)
        writer.close()  # idempotent: a second close is a no-op

    def test_early_encoder_exit_surfaces_as_an_error(self):
        make_script(self.root / "bin" / "ffmpeg", EXIT_FFMPEG)
        writer = cli.SoftwareVideoWriter(self.out, self.audio, 16, 12)
        writer.STALL_TIMEOUT = 10
        # Frames far larger than the pipe + queue keep submit in flight when
        # the encoder exits, so the failure must surface instead of blocking.
        blob = b"x" * (1 << 18)
        with self.assertRaises(RuntimeError):
            for index in range(10):
                writer.submit(blob)
        writer.abort()
        self.assertTrue(writer.log.closed)

    def test_stalled_encoder_times_out_and_aborts(self):
        make_script(self.root / "bin" / "ffmpeg", STALL_FFMPEG)
        writer = cli.SoftwareVideoWriter(self.out, self.audio, 16, 12)
        writer.STALL_TIMEOUT = 0.5
        blob = b"x" * (1 << 18)
        with self.assertRaises(TimeoutError):
            for index in range(10):
                writer.submit(blob)
        writer.abort()
        self.assertTrue(writer.log.closed)
        self.assertFalse(writer.thread.is_alive())
        # The parent's pipe write-end is closed too, not just the log.
        self.assertTrue(writer.proc.stdin.closed)
        # The whole encoder process group is gone: the non-exec ``sleep``
        # child cannot survive and hold the pipe read-end open.
        self._assert_group_reaped(writer)

    def _assert_group_reaped(self, writer):
        """No live member remains in the encoder's process group.

        A killed grandchild may linger as a zombie until the init process
        reaps it — not a leak: the pipe read-end closed when the process
        died. Zombies still answer ``killpg``, so on Linux (``/proc``
        available) zombie members are ignored; on macOS the host init
        reaps adopted orphans promptly and ``killpg`` is exact.
        """

        import time as _time
        from pathlib import Path as _Path

        def group_has_live_members():
            try:
                os.killpg(writer.proc.pid, 0)
            except ProcessLookupError:
                return False
            # The group answered; a zombie answering is not a live member.
            for stat in _Path("/proc").glob("[0-9]*/stat"):
                try:
                    _, rest = stat.read_text().rsplit(")", 1)
                    fields = rest.split()
                    # stat layout after (comm): state ppid pgrp session ...
                    if (int(fields[2]) == writer.proc.pid
                            and fields[0] != "Z"):
                        return True
                except (OSError, ValueError, IndexError):
                    continue  # /proc entry vanished mid-scan
            return False

        for _ in range(50):  # init reaping can lag a moment
            if not group_has_live_members():
                return
            _time.sleep(0.1)
        self.fail(
            f"process group {writer.proc.pid} still has live members "
            "after abort"
        )

    def test_constructor_failure_leaves_no_log_file(self):
        with mock.patch.dict(os.environ, {"PATH": "/nonexistent-dir-only"}):
            with self.assertRaises(FileNotFoundError):
                cli.SoftwareVideoWriter(self.out, self.audio, 16, 12)
        self.assertFalse(Path(f"{self.out}.ffmpeg.log").exists())

    def test_make_writer_requires_ffmpeg_on_path(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                cli.make_writer("software", self.out, self.audio, 16, 12)


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
