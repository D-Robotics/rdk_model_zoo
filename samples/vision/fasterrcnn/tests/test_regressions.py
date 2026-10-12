# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Regressions for the live-review defects (2026-10-12).

Each test pins the exact semantics against an independently computed
reference — never the implementation's own formula:

* ``clip_boxes_to_image`` must follow TorchVision: operate on the last axis
  for any leading shape (including the ``(N, 91, 4)`` tail arrays) and
  clamp continuous coordinates to ``width``/``height``, not ``size - 1``.
* ``LetterboxContext.to_original`` must invert the *achieved* per-axis
  resize (``net / original``), not the nominal scalar scale; rounding can
  make the two differ by a full pixel at content borders.
* The NV12 input must encode the BGR canvas through the shared converter
  (limited-range BT.601); converting to RGB first swaps red and blue.
* The historical ``pre_process``/``forward``/``post_process`` aliases must
  stay out of this new sample's task classes.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.fasterrcnn.runtime.python import detect as D  # noqa: E402


class ClipBoxesRegressions(unittest.TestCase):
    def test_codex_repro_3d_two_identical_boxes(self):
        # (1, 2, 4): both rows must clip identically to [0, w] x [0, h].
        boxes = np.array([[[0.0, 0.0, 400.0, 500.0],
                           [0.0, 0.0, 400.0, 500.0]]], dtype=np.float32)
        got = D.clip_boxes_to_image(boxes, 200, 300)
        expected = np.array([[[0.0, 0.0, 300.0, 200.0],
                              [0.0, 0.0, 300.0, 200.0]]], dtype=np.float32)
        np.testing.assert_array_equal(got, expected)

    def test_2d_clamps_to_full_extent(self):
        got = D.clip_boxes_to_image(np.array([[0.0, 0.0, 400.0, 500.0]], dtype=np.float32),
                                    200, 300)
        np.testing.assert_array_equal(got, np.array([[0.0, 0.0, 300.0, 200.0]], dtype=np.float32))

    def test_negative_coordinates_clamp_to_zero(self):
        got = D.clip_boxes_to_image(np.array([[-5.0, -3.0, 50.0, 60.0]], dtype=np.float32),
                                    100, 100)
        np.testing.assert_array_equal(got, np.array([[0.0, 0.0, 50.0, 60.0]], dtype=np.float32))

    def test_square_canvas_border_box_reaches_extent(self):
        got = D.clip_boxes_to_image(np.array([[-5.0, -5.0, 330.0, 340.0]], dtype=np.float32),
                                    320, 320)
        np.testing.assert_array_equal(got, np.array([[0.0, 0.0, 320.0, 320.0]], dtype=np.float32))

    def test_in_range_boxes_untouched(self):
        boxes = np.array([[1.5, 2.5, 318.5, 319.5], [0.0, 0.0, 320.0, 320.0]], dtype=np.float32)
        np.testing.assert_array_equal(D.clip_boxes_to_image(boxes, 320, 320), boxes)

    def test_3d_per_class_rows_clip_independently(self):
        # The (N, K, 4) tail shape: every (proposal, class) row is clipped
        # by the same bounds; leading axes are never mixed.
        boxes = np.array([[[400.0, 0.0, 500.0, 100.0], [0.0, 500.0, 100.0, 600.0]],
                          [[-10.0, -10.0, 10.0, 10.0], [1.0, 1.0, 2.0, 2.0]]], dtype=np.float32)
        got = D.clip_boxes_to_image(boxes, 200, 300)
        expected = np.array([[[300.0, 0.0, 300.0, 100.0], [0.0, 200.0, 100.0, 200.0]],
                             [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 2.0, 2.0]]], dtype=np.float32)
        np.testing.assert_array_equal(got, expected)

    def test_tail_path_clips_per_class(self):
        # End-to-end shape the tail uses: decode produces (N, 91, 4); after
        # clipping, background removal and flattening must stay aligned.
        n = 3
        boxes = np.zeros((n, 91, 4), dtype=np.float32)
        boxes[:, :, 2] = 1000.0   # every class row overshoots horizontally
        boxes[:, :, 3] = 1000.0
        got = D.clip_boxes_to_image(boxes, 320, 320)
        self.assertEqual(got.shape, (n, 91, 4))
        self.assertTrue((got[..., 2] == 320.0).all())
        self.assertTrue((got[..., 3] == 320.0).all())


class LetterboxInverseRegressions(unittest.TestCase):
    def test_wide_image_bottom_and_right_edges(self):
        # 161x640: nominal scale 0.5, but net height rounds 80.5 -> 80
        # (half-to-even), so the achieved y factor is 80/161, not 0.5.
        # Canvas right edge x=320, content bottom y=80.
        context = D.letterbox_geometry(161, 640)
        self.assertEqual(context.net_size, (80, 320))
        out = context.to_original(np.array([[320.0, 80.0, 320.0, 80.0]], dtype=np.float32))
        self.assertAlmostEqual(out[0, 0], 640.0, places=4)   # 320 / (320/640)
        self.assertAlmostEqual(out[0, 2], 640.0, places=4)
        self.assertAlmostEqual(out[0, 1], 161.0, places=4)   # 80 / (80/161)
        self.assertAlmostEqual(out[0, 3], 161.0, places=4)

    def test_tall_image_right_and_bottom_edges(self):
        # 640x161: content right edge x=80, canvas bottom y=320.
        context = D.letterbox_geometry(640, 161)
        self.assertEqual(context.net_size, (320, 80))
        out = context.to_original(np.array([[80.0, 320.0, 80.0, 320.0]], dtype=np.float32))
        self.assertAlmostEqual(out[0, 0], 161.0, places=4)   # 80 / (80/161)
        self.assertAlmostEqual(out[0, 2], 161.0, places=4)
        self.assertAlmostEqual(out[0, 1], 640.0, places=4)   # 320 / (320/640)
        self.assertAlmostEqual(out[0, 3], 640.0, places=4)

    def test_uneven_scale_axes(self):
        # 319x100: nominal scale 320/319 (upscale).  net_w = round(100 *
        # 320/319) = 100, so the achieved x factor is exactly 1.0, while
        # the y factor is 320/319: content bottom y=320 maps to 319.
        context = D.letterbox_geometry(319, 100)
        self.assertEqual(context.net_size, (320, 100))
        self.assertAlmostEqual(context.scale_x, 1.0, places=6)
        self.assertAlmostEqual(context.scale_y, 320.0 / 319.0, places=6)
        out = context.to_original(np.array([[100.0, 320.0, 100.0, 320.0]], dtype=np.float32))
        self.assertAlmostEqual(out[0, 0], 100.0, places=4)   # x axis is identity
        self.assertAlmostEqual(out[0, 2], 100.0, places=4)
        self.assertAlmostEqual(out[0, 1], 319.0, places=4)   # 320 / (320/319)
        self.assertAlmostEqual(out[0, 3], 319.0, places=4)

    def test_square_identity(self):
        context = D.letterbox_geometry(320, 320)
        boxes = np.array([[3.0, 4.0, 300.0, 301.0]], dtype=np.float32)
        np.testing.assert_allclose(context.to_original(boxes), boxes, atol=1e-6)

    def test_out_of_range_canvas_boxes_clip_to_source(self):
        context = D.letterbox_geometry(161, 640)
        out = context.to_original(np.array([[0.0, 0.0, 320.0, 320.0]], dtype=np.float32))
        self.assertEqual(out[0, 2], 640.0)
        self.assertEqual(out[0, 3], 161.0)

    def test_scale_properties_report_achieved_factors(self):
        context = D.letterbox_geometry(161, 640)
        self.assertAlmostEqual(context.scale_y, 80.0 / 161.0, places=6)
        self.assertAlmostEqual(context.scale_x, 320.0 / 640.0, places=6)


def bt601_limited(b: int, g: int, r: int) -> tuple[int, int, int]:
    """Independent reference: full-range BT.601 rescaled to limited range.

    Computes (Y, U, V) with the standard matrix and the studio-swing
    rescaling (Y: 16..235 via 219/255, U/V: 16..240 via 224/255 around 128)
    — the convention the compiled PartA input declares as bt601_video.
    """
    y_full = 0.299 * r + 0.587 * g + 0.114 * b
    u_full = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    v_full = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    y = round(16.0 + y_full * 219.0 / 255.0)
    u = round(128.0 + (u_full - 128.0) * 224.0 / 255.0)
    v = round(128.0 + (v_full - 128.0) * 224.0 / 255.0)
    return int(min(235, max(16, y))), int(min(240, max(16, u))), int(min(240, max(16, v)))


def solid_canvas(b: int, g: int, r: int, size: int = 320) -> np.ndarray:
    """A solid-color BGR canvas (cv2's channel order)."""
    return np.tile(np.array([b, g, r], dtype=np.uint8), (size, size, 1))


class Nv12ColorRegressions(unittest.TestCase):
    """The NV12 input must be built from BGR pixels (limited-range BT.601)."""

    COLORS = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }

    def _yuv(self, tensors):
        y = tensors["images_y"]
        uv = tensors["images_uv"]
        return int(y[0, 0, 0, 0]), int(uv[0, 0, 0, 0]), int(uv[0, 0, 0, 1])

    def test_solid_colors_match_independent_bt601(self):
        for name, (b, g, r) in self.COLORS.items():
            with self.subTest(color=name):
                y, u, v = self._yuv(D.nv12_tensors(solid_canvas(b, g, r), "s100"))
                ey, eu, ev = bt601_limited(b, g, r)
                self.assertAlmostEqual(y, ey, delta=1)
                self.assertAlmostEqual(u, eu, delta=1)
                self.assertAlmostEqual(v, ev, delta=1)

    def test_red_is_not_blue(self):
        # A channel swap turns pure red into pure blue: distinct luma and a
        # distinct (Y, U, V) triple prove the BGR contract is honored.  The
        # symmetric chroma pair V(red) == U(blue) both saturate at 240.
        red = self._yuv(D.nv12_tensors(solid_canvas(*self.COLORS["red"]), "s100"))
        blue = self._yuv(D.nv12_tensors(solid_canvas(*self.COLORS["blue"]), "s100"))
        self.assertNotEqual(red, blue)
        self.assertNotEqual(red[0], blue[0])
        self.assertEqual(red[2], blue[1])   # V(red) == U(blue) == 240 (saturated)

    def test_rgb_reordered_canvas_is_rejected_by_reference(self):
        # Feeding an RGB-ordered array to the BGR converter reproduces the
        # original defect: the red canvas encodes as blue.
        import cv2

        b, g, r = self.COLORS["red"]
        swapped = cv2.cvtColor(solid_canvas(b, g, r), cv2.COLOR_BGR2RGB)
        encoded = self._yuv(D.nv12_tensors(swapped, "s100"))
        blue = self._yuv(D.nv12_tensors(solid_canvas(*self.COLORS["blue"]), "s100"))
        self.assertEqual(encoded, blue)

    def test_x5_flat_matches_s_split_bytes(self):
        for name, (b, g, r) in self.COLORS.items():
            with self.subTest(color=name):
                flat = D.nv12_tensors(solid_canvas(b, g, r), "x5")["images"]
                split = D.nv12_tensors(solid_canvas(b, g, r), "s100")
                self.assertEqual(flat.dtype, np.uint8)
                self.assertEqual(
                    flat.tobytes(),
                    np.concatenate((split["images_y"].reshape(-1),
                                    split["images_uv"].reshape(-1))).tobytes())

    def test_nv12_plane_shapes(self):
        tensors = D.nv12_tensors(solid_canvas(0, 0, 0), "s600")
        self.assertEqual(tensors["images_y"].shape, (1, 320, 320, 1))
        self.assertEqual(tensors["images_uv"].shape, (1, 160, 160, 2))
        self.assertEqual(D.nv12_tensors(solid_canvas(0, 0, 0), "x5")["images"].size,
                         320 * 320 * 3 // 2)

    def test_invalid_canvas_rejected(self):
        with self.assertRaises(ValueError):
            D.nv12_tensors(np.zeros((4, 4, 3), dtype=np.float32), "x5")
        with self.assertRaises(ValueError):
            D.nv12_tensors(np.zeros((4, 4), dtype=np.uint8), "x5")


class RemovedAliasRegressions(unittest.TestCase):
    def test_task_has_no_historical_aliases(self):
        for name in ("pre_process", "forward", "post_process"):
            self.assertFalse(hasattr(D.FasterRCNNTask, name), name)

    def test_tracker_has_no_historical_aliases(self):
        from samples.vision.fasterrcnn.runtime.python.tracking import PersonTrackerTask

        for name in ("pre_process", "forward", "post_process"):
            self.assertFalse(hasattr(PersonTrackerTask, name), name)


class _FakeTrackStream:
    """Minimal tracker contract for ``track_video`` (empty frames, no SDK)."""

    def __init__(self):
        self.last_tracker_us = 0.0
        self.detector = type("Detector", (), {"last_proposals": 0})()

    def predict(self, frame):
        return []

    def reset(self):
        pass


class TrackVideoIORegressions(unittest.TestCase):
    """The annotated-video writer must be honest (2026-10-12 CLI review).

    A two-frame video plus an empty fake tracker reproduced: the run
    reported ``frames: 2`` and a result path under a directory that was
    never created, and no file appeared.  Parents must be created, the
    writer must open before the first frame, failures must raise, and
    ``--no-encode`` must skip the writer without claiming an output.
    """

    VIDEO_SIZE = (64, 48)

    @classmethod
    def setUpClass(cls):
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cv2 not available for the video IO tests")
        cls._wrote_video = False

    def _make_video(self, directory, frames=2):
        import cv2

        path = directory / "input.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                                 self.VIDEO_SIZE)
        if not writer.isOpened():
            self.skipTest("mp4v encoding unavailable on this host")
        for i in range(frames):
            writer.write(np.full((self.VIDEO_SIZE[1], self.VIDEO_SIZE[0], 3),
                                 30 * (i + 1), dtype=np.uint8))
        writer.release()
        return path

    def test_output_parent_directories_created(self):
        import tempfile
        from samples.vision.fasterrcnn.runtime.python.cli import track_video

        with tempfile.TemporaryDirectory() as tmp:
            video = self._make_video(Path(tmp))
            out = Path(tmp) / "new" / "nested" / "out.mp4"
            summary = track_video(_FakeTrackStream(), video, out)
            self.assertTrue(out.is_file(), out)
            self.assertEqual(summary["frames"], 2)
            self.assertEqual(summary["result_path"], str(out))

    def test_unwritable_output_raises_before_processing(self):
        import tempfile
        from samples.vision.fasterrcnn.runtime.python.cli import track_video

        with tempfile.TemporaryDirectory() as tmp:
            video = self._make_video(Path(tmp))
            blocker = Path(tmp) / "blocker.mp4"
            blocker.mkdir()  # a directory cannot be a writer target
            with self.assertRaises(OSError):
                track_video(_FakeTrackStream(), video, blocker)

    def test_no_encode_skips_writer_and_reports_null(self):
        import tempfile
        from samples.vision.fasterrcnn.runtime.python.cli import track_video

        with tempfile.TemporaryDirectory() as tmp:
            video = self._make_video(Path(tmp))
            out = Path(tmp) / "skipped" / "out.mp4"
            summary = track_video(_FakeTrackStream(), video, out, encode=False)
            self.assertEqual(summary["frames"], 2)
            self.assertIsNone(summary["result_path"])
            self.assertFalse(out.exists())
            self.assertFalse(out.parent.is_dir(), "no parent should be created")

    def test_relative_output_parent_resolves_against_cwd(self):
        import os
        import tempfile
        from samples.vision.fasterrcnn.runtime.python.cli import track_video

        with tempfile.TemporaryDirectory() as tmp:
            video = self._make_video(Path(tmp))
            # contextlib.chdir needs Python 3.11; restore the cwd by hand so
            # the suite also runs on the 3.10 CI matrix.
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                summary = track_video(_FakeTrackStream(), video, Path("rel/out.mp4"))
            finally:
                os.chdir(old_cwd)
            self.assertTrue((Path(tmp) / "rel" / "out.mp4").is_file())
            self.assertEqual(summary["result_path"], "rel/out.mp4")


if __name__ == "__main__":
    unittest.main()
