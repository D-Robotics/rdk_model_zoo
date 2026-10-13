"""Face-generator support pieces: LZ4 cache, ring order, renderer I/O.

The renderer is exercised against real avatar data directories (reference
JPEGs, face box) with the hardware-decoded background frames injected
through ``AvatarRenderer(background_frames=...)`` — the same list
``media_worker.decode_background`` returns on the board — and FakeRuntime
stages; compositing is checked against the documented activation/blend
arithmetic.  OpenCV appears only as an independent host-test oracle (the
runtime itself is cv2-free); its measured tolerances are pinned per check.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from samples.vision.lite_avatar.runtime.python import avatar
from samples.vision.lite_avatar.tests.doubles import (
    FakeRuntime,
    build_avatar_data,
)

CMVN = None  # the frontend is not needed in this module


class CompressedBackgroundTests(unittest.TestCase):
    def test_roundtrip_is_lossless_and_bounded(self):
        frame = np.tile(
            np.arange(0, 255, 4, dtype=np.uint8), (32, 12, 3)
        )[:, :48, :]
        item = avatar.CompressedBackground(frame)
        self.assertEqual((item.shape, str(item.dtype)), (frame.shape, str(frame.dtype)))
        self.assertEqual(item.frame_shape, frame.shape)
        self.assertLess(item.storage_nbytes, frame.nbytes)
        back = item.copy()
        self.assertTrue(np.array_equal(back, frame))
        # Decompressing into a caller buffer avoids per-frame allocation.
        reused = np.empty_like(frame)
        self.assertIs(item.copy(out=reused), reused)
        self.assertTrue(np.array_equal(reused, frame))

    def test_rejects_incompatible_output_buffers(self):
        item = avatar.CompressedBackground(np.zeros((8, 8, 3), np.uint8))
        for bad in (
            np.zeros((8, 8, 4), np.uint8),
            np.zeros((7, 8, 3), np.uint8),
            np.zeros((8, 8, 3), np.float32),
        ):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                item.copy(out=bad)


class RingOrderTests(unittest.TestCase):
    def test_ping_pong_cycle_dwells_at_both_ends(self):
        count = 3
        order = [avatar.background_index_for(i, count) for i in range(12)]
        self.assertEqual(order, [0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0])

    def test_stays_inside_the_ring_for_any_index(self):
        for count in (1, 5, 150):
            for index in (0, count - 1, count, 7 * count + 3, 100_000):
                value = avatar.background_index_for(index, count)
                self.assertTrue(0 <= value < count, (index, count, value))


class ResizeAndReferenceTests(unittest.TestCase):
    """cv2-free image helpers, pinned against OpenCV as an oracle.

    ``resize_bilinear`` reproduces ``cv2.resize(..., INTER_LINEAR)`` to a
    maximum deviation of 1 (per-pixel rounding), measured across square,
    non-square and the sample's real geometries; Pillow JPEG decode was
    measured bit-identical to ``cv2.imread`` over all 150 references.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_identity_and_upscale_zero_are_exact(self):
        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, (9, 13, 3), dtype=np.uint8)
        np.testing.assert_array_equal(
            avatar.resize_bilinear(image, (13, 9)), image
        )
        # An exact 2x upscale of a checkerboard corner stays exact at the
        # sample corners and never invents values outside the source range.
        small = np.array([[[10, 20, 30], [200, 0, 255]]], np.uint8)
        up = avatar.resize_bilinear(small, (4, 2))
        self.assertEqual(up.shape, (2, 4, 3))
        in_range = (up.astype(int) >= small.min()) & (up.astype(int) <= small.max())
        self.assertTrue(in_range.all())

    def test_matches_opencv_inter_linear_within_one(self):
        import cv2

        rng = np.random.default_rng(11)
        for src_shape, size in (
            ((48, 64), (32, 24)),  # downscale, square-ish
            ((1920, 890), (384, 384)),  # reference encode geometry
            ((384, 384), (419, 289)),  # mouth ROI geometry
            ((37, 53), (96, 20)),  # upscale, non-square
        ):
            with self.subTest(src_shape=src_shape, size=size):
                image = rng.integers(0, 256, src_shape + (3,), dtype=np.uint8)
                got = avatar.resize_bilinear(image, size)
                want = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
                deviation = np.abs(got.astype(int) - want.astype(int))
                self.assertLessEqual(int(deviation.max()), 1)

    def test_load_reference_bgr_matches_opencv_imread_bitwise(self):
        import cv2

        rng = np.random.default_rng(13)
        path = self.root / "ref.jpg"
        cv2.imwrite(str(path), rng.integers(0, 256, (40, 44, 3), dtype=np.uint8))
        np.testing.assert_array_equal(
            avatar.load_reference_bgr(path), cv2.imread(str(path))
        )


class NormalizeReferenceTests(unittest.TestCase):
    def test_maps_pixels_into_minus_one_one_rgb(self):
        image = np.zeros((100, 100, 3), np.uint8)
        image[:50] = (255, 0, 0)  # BGR blue
        image[50:] = (255, 255, 255)  # white
        tensor = avatar.normalize_reference(image)
        self.assertEqual(tensor.shape, (1, 3, 384, 384))
        self.assertEqual(tensor.dtype, np.float32)
        center = tensor[0, :, 100, 192]
        np.testing.assert_allclose(center, [-1.0, -1.0, 1.0], atol=1e-6)
        bottom = tensor[0, :, 300, 192]
        np.testing.assert_allclose(bottom, [1.0, 1.0, 1.0], atol=1e-6)
        self.assertTrue(np.isfinite(tensor).all())
        # A 4-channel image is reduced to its first three channels first.
        bgra = np.dstack([image, np.full((100, 100), 9, np.uint8)])
        np.testing.assert_array_equal(
            avatar.normalize_reference(bgra), tensor
        )


def facegen_stages(decoder_value=0.5, encoder_log=None):
    """Encoder/decoder stages over FakeRuntimes with constant outputs."""

    def encoder_transform(feed):
        if encoder_log is not None:
            encoder_log.append(1)
        shapes = (
            (1, 3, 384, 384),
            (1, 16, 192, 192),
            (1, 32, 96, 96),
            (1, 64, 48, 48),
        )
        return {f"f{i}": np.zeros(s, np.float32) for i, s in enumerate(shapes)}

    def decoder_transform(feed):
        return {
            "mouth_image": np.full((1, 3, 384, 384), decoder_value, np.float32)
        }

    encoder = avatar.FaceGenEncoderStage(
        FakeRuntime("facegen_encoder", transform=encoder_transform),
        "image",
        ("f0", "f1", "f2", "f3"),
        None,
    )
    decoder = avatar.FaceGenDecoderStage(
        FakeRuntime("facegen_decoder", transform=decoder_transform),
        ("f0", "f1", "f2", "f3"),
        "mouth_param",
        "mouth_image",
        None,
    )
    return encoder, decoder


class OwnedList(list):
    """Plain ``list`` admits no weak references; this subclass does.

    It exists only so the memory-contract tests can observe when the
    renderer's last strong reference to a background list goes away —
    the renderer treats it as any other list.
    """


class AvatarRendererTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _renderer(self, max_refs=150, **build_kwargs):
        encoder, decoder = facegen_stages()
        data, frames = build_avatar_data(self.root / "avatar", **build_kwargs)
        return avatar.AvatarRenderer(
            data, encoder, decoder, max_refs=max_refs, background_frames=frames
        )

    def test_initializes_the_ring_box_and_reference_cache(self):
        renderer = self._renderer()
        self.assertEqual(len(renderer.bg), 6)
        self.assertEqual(renderer.bg_n, 6)
        self.assertEqual(len(renderer.ref_feats), 6)
        self.assertEqual((renderer.width, renderer.height), (64, 48))
        self.assertGreater(renderer.cache_mib, 0)

    def test_composites_the_decoded_mouth_over_the_background(self):
        import cv2

        encoder, decoder = facegen_stages(decoder_value=0.5)
        data, frames = build_avatar_data(self.root / "avatar")
        renderer = avatar.AvatarRenderer(
            data, encoder, decoder, background_frames=frames
        )
        frame = renderer.gen_frame(np.full((1, 32), 0.25, np.float32), 1)
        self.assertEqual(frame.shape, (48, 64, 3))
        self.assertEqual(frame.dtype, np.uint8)
        # The activation is 0.5*0.5+0.5 = 0.75 -> 0.75*255 = 191.25 -> 191;
        # the resize of a constant stays constant.
        background = renderer.bg[1].copy()
        # Outside the box the background frame is untouched.
        np.testing.assert_array_equal(frame[0, 0], background[0, 0])
        np.testing.assert_array_equal(frame[-1, -1], background[-1, -1])
        # Inside the box the blend is mouth * inverse_mask + roi * mask with
        # the source's feather mask, reconstructed independently here from
        # cv2.GaussianBlur (the runtime uses the scipy "mirror" equivalent;
        # measured max deviation 0.0043 -> <=2/255 in the blended output).
        mask = (np.ones((renderer.y2 - renderer.y1, renderer.x2 - renderer.x1, 3)) * 255).astype(np.uint8)
        mask[10:-10, 10:-10, :] *= 0
        mask = (cv2.GaussianBlur(mask, (21, 21), 15) / 255.0).astype(np.float32)
        mouth = np.full(
            (renderer.y2 - renderer.y1, renderer.x2 - renderer.x1, 3), 191, np.uint8
        )
        roi = background[renderer.y1 : renderer.y2, renderer.x1 : renderer.x2, :]
        blended = (mouth * (1 - mask) + roi * mask).astype(np.uint8)
        np.testing.assert_allclose(
            frame[renderer.y1 : renderer.y2, renderer.x1 : renderer.x2, :],
            blended,
            atol=2,
        )
        # Sanity anchors: the ROI center sits strictly between the mouth
        # constant and the background value it moderates.
        center = frame[
            (renderer.y1 + renderer.y2) // 2, (renderer.x1 + renderer.x2) // 2
        ].astype(int)
        self.assertTrue((center < 191).all())
        self.assertTrue((center > background[20, 30].astype(int) - 5).all())

    def test_gen_frame_can_render_into_a_caller_buffer(self):
        renderer = self._renderer()
        buffer = np.empty((48, 64, 3), np.uint8)
        out = renderer.gen_frame(np.zeros((1, 32), np.float32), 0, out=buffer)
        self.assertIs(out, buffer)
        self.assertTrue((out >= 0).all())

    def test_reference_pyramids_are_encoded_once(self):
        encoder_log = []
        encoder, decoder = facegen_stages(encoder_log=encoder_log)
        data, frames = build_avatar_data(self.root / "avatar")
        renderer = avatar.AvatarRenderer(
            data, encoder, decoder, background_frames=frames
        )
        encoded = len(encoder_log)
        renderer.gen_frame(np.zeros((1, 32), np.float32), 2)
        renderer.gen_frame(np.zeros((1, 32), np.float32), 3)
        self.assertEqual(len(encoder_log), encoded)

    def test_rejects_broken_avatar_data(self):
        encoder, decoder = facegen_stages()

        # No background frames at all (the injection seam carries the empty
        # hardware decode result).
        empty = self.root / "empty"
        (empty / "ref_frames").mkdir(parents=True)
        (empty / "face_box.txt").write_text("10 38 12 44\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(empty, encoder, decoder, background_frames=[])

        # Box outside the frame.
        outside, outside_frames = build_avatar_data(
            self.root / "outside", box=(10, 60, 12, 70)
        )
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(
                outside, encoder, decoder, background_frames=outside_frames
            )

        # Box smaller than the feather band.
        tiny, tiny_frames = build_avatar_data(self.root / "tiny", box=(10, 28, 12, 30))
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(tiny, encoder, decoder, background_frames=tiny_frames)

        # Non-integer box values.
        badbox, badbox_frames = build_avatar_data(self.root / "badbox")
        (badbox / "face_box.txt").write_text("a b c d\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(badbox, encoder, decoder, background_frames=badbox_frames)

        # Undecodable reference image.
        garbage, garbage_frames = build_avatar_data(self.root / "garbage")
        (garbage / "ref_frames" / "ref_00003.jpg").write_bytes(b"not a jpeg")
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(garbage, encoder, decoder, background_frames=garbage_frames)

        # Fewer references than ring slots.
        few, few_frames = build_avatar_data(self.root / "few", refs=3)
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(few, encoder, decoder, background_frames=few_frames)

        # No reference frames at all.
        norefs, norefs_frames = build_avatar_data(self.root / "norefs", refs=0)
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(norefs, encoder, decoder, background_frames=norefs_frames)

        # Mismatched background geometries are rejected (one geometry only).
        mixed = self.root / "mixed"
        _, mixed_frames = build_avatar_data(mixed)
        mixed_frames[3] = mixed_frames[3][:, :32]
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(mixed, encoder, decoder, background_frames=mixed_frames)

        # Extra references beyond the ring are not even decoded: exactly
        # bg_n pyramids are encoded.
        extra, extra_frames = build_avatar_data(self.root / "extra", refs=8)
        renderer = avatar.AvatarRenderer(
            extra, encoder, decoder, background_frames=extra_frames
        )
        self.assertEqual(renderer.bg_n, 6)
        self.assertEqual(len(renderer.ref_feats), 6)

    def test_caches_are_bounded_to_the_ring_slots_actually_used(self):
        renderer = self._renderer(max_refs=4)
        self.assertEqual(len(renderer.bg), 4)
        self.assertEqual(renderer.bg_n, 4)
        self.assertEqual(len(renderer.ref_feats), 4)

    def test_injected_background_frames_are_never_retained_or_mutated(self):
        # The caller owns injected frames: the compressed ring holds copies
        # as payload bytes, so once the test lets go of the list nothing in
        # the renderer keeps it alive, and rendering never mutates it.
        import weakref

        encoder, decoder = facegen_stages()
        data, _ = build_avatar_data(self.root / "avatar")
        frames = OwnedList(
            np.full((48, 64, 3), value, np.uint8)
            for value in (30, 60, 90, 120, 150, 180)
        )
        copies = [frame.copy() for frame in frames]
        owned = weakref.ref(frames)
        renderer = avatar.AvatarRenderer(
            data, encoder, decoder, background_frames=frames
        )
        renderer.gen_frame(np.zeros((1, 32), np.float32), 0)
        self.assertEqual(len(frames), len(copies))
        for got, want in zip(frames, copies):
            np.testing.assert_array_equal(got, want)
        del frames, got, want
        self.assertIsNone(
            owned(), "renderer must not retain the caller's background list"
        )

    def test_decoded_background_list_is_released_before_the_pyramids_load(self):
        # Memory contract of the board path: once the ring holds the
        # compressed payloads, the raw decoded list (~733 MiB for the real
        # 150-frame 890x1920 asset) must be released *before* the reference
        # pyramids load.  The list is created inside a side_effect factory —
        # mock.return_value would retain it — so the weak reference is the
        # only handle left, and it must already be dead at the first
        # load_reference_bgr call (this fails on code that keeps the decode
        # result alive through __init__, which is the regression it pins).
        import weakref
        from unittest import mock

        encoder, decoder = facegen_stages()
        data, _ = build_avatar_data(self.root / "avatar")
        owned = []

        def decode_factory(data_dir, max_frames=150):
            frames = OwnedList(
                np.full((48, 64, 3), value, np.uint8)
                for value in (30, 60, 90, 120, 150, 180)
            )
            owned.append(weakref.ref(frames))
            return frames

        real_load = avatar.load_reference_bgr
        loads = []

        def probed_load(path):
            loads.append(path)
            if len(loads) == 1:
                self.assertIsNone(
                    owned[0](),
                    "raw decoded background list is still alive when the "
                    "first reference pyramid load begins",
                )
            return real_load(path)

        with mock.patch(
            "samples.vision.lite_avatar.runtime.python.media_worker.decode_background",
            side_effect=decode_factory,
        ), mock.patch.object(
            avatar, "load_reference_bgr", side_effect=probed_load
        ):
            renderer = avatar.AvatarRenderer(data, encoder, decoder)
        self.assertEqual(len(renderer.bg), 6)  # compression happened
        self.assertGreaterEqual(len(loads), 1)  # the probe really observed it
        self.assertIsNone(owned[0]())  # and the list stays dead afterwards

    def test_default_background_load_requires_the_prepared_asset(self):
        # With no injected frames the renderer loads the background through
        # the hardware path's sidecar validation; a data directory holding
        # only the legacy MPEG-4 asset explains the one-time host
        # preparation instead of decoding (cv2 writes the legacy file here
        # as test fixture only — the runtime never reads it).
        import cv2

        encoder, decoder = facegen_stages()
        legacy = self.root / "legacy"
        legacy.mkdir()
        cv2.VideoWriter(
            str(legacy / "bg_video.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (64, 48),
        ).release()
        with self.assertRaises(ValueError) as raised:
            avatar.AvatarRenderer(legacy, encoder, decoder)
        message = str(raised.exception)
        self.assertIn("bg_video.mp4", message)
        self.assertIn("ffmpeg", message)
        self.assertNotIn("hobot_vio", message)


if __name__ == "__main__":
    unittest.main()
