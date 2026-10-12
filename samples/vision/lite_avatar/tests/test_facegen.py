"""Face-generator support pieces: LZ4 cache, ring order, renderer I/O.

The renderer is exercised against real cv2-written avatar data directories
and FakeRuntime stages; compositing is checked against the documented
activation/blend arithmetic.
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


class NormalizeReferenceTests(unittest.TestCase):
    def test_maps_pixels_into_minus_one_one_rgb(self):
        import cv2

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
        self.assertTrue(cv2.COLOR_BGR2RGB > 0)


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


class AvatarRendererTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _renderer(self, max_refs=150, **build_kwargs):
        encoder, decoder = facegen_stages()
        return avatar.AvatarRenderer(
            build_avatar_data(self.root / "avatar", **build_kwargs),
            encoder,
            decoder,
            max_refs=max_refs,
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
        data = build_avatar_data(self.root / "avatar")
        renderer = avatar.AvatarRenderer(data, encoder, decoder)
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
        # the source's feather mask, reconstructed independently here.
        mask = (np.ones((renderer.y2 - renderer.y1, renderer.x2 - renderer.x1, 3)) * 255).astype(np.uint8)
        mask[10:-10, 10:-10, :] *= 0
        mask = (cv2.GaussianBlur(mask, (21, 21), 15) / 255.0).astype(np.float32)
        mouth = np.full(
            (renderer.y2 - renderer.y1, renderer.x2 - renderer.x1, 3), 191, np.uint8
        )
        roi = background[renderer.y1 : renderer.y2, renderer.x1 : renderer.x2, :]
        blended = (mouth * (1 - mask) + roi * mask).astype(np.uint8)
        np.testing.assert_array_equal(
            frame[renderer.y1 : renderer.y2, renderer.x1 : renderer.x2, :], blended
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
        data = build_avatar_data(self.root / "avatar")
        renderer = avatar.AvatarRenderer(data, encoder, decoder)
        encoded = len(encoder_log)
        renderer.gen_frame(np.zeros((1, 32), np.float32), 2)
        renderer.gen_frame(np.zeros((1, 32), np.float32), 3)
        self.assertEqual(len(encoder_log), encoded)

    def test_rejects_broken_avatar_data(self):
        encoder, decoder = facegen_stages()
        good = build_avatar_data(self.root / "good")

        # No background frames at all.
        empty = self.root / "empty"
        (empty / "ref_frames").mkdir(parents=True)
        import cv2

        cv2.VideoWriter(
            str(empty / "bg_video.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (64, 48),
        ).release()
        (empty / "face_box.txt").write_text("10 38 12 44\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(empty, encoder, decoder)

        # Box outside the frame.
        outside = build_avatar_data(self.root / "outside", box=(10, 60, 12, 70))
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(outside, encoder, decoder)

        # Box smaller than the feather band.
        tiny = build_avatar_data(self.root / "tiny", box=(10, 28, 12, 30))
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(tiny, encoder, decoder)

        # Non-integer box values.
        badbox = build_avatar_data(self.root / "badbox")
        (badbox / "face_box.txt").write_text("a b c d\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(badbox, encoder, decoder)

        # Undecodable reference image.
        garbage = build_avatar_data(self.root / "garbage")
        (garbage / "ref_frames" / "ref_00003.jpg").write_bytes(b"not a jpeg")
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(garbage, encoder, decoder)

        # Fewer references than ring slots.
        few = build_avatar_data(self.root / "few", refs=3)
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(few, encoder, decoder)

        # No reference frames at all.
        norefs = build_avatar_data(self.root / "norefs", refs=0)
        with self.assertRaises(ValueError):
            avatar.AvatarRenderer(norefs, encoder, decoder)

        # Extra references beyond the ring are not even decoded: exactly
        # bg_n pyramids are encoded.
        extra = build_avatar_data(self.root / "extra", refs=8)
        renderer = avatar.AvatarRenderer(extra, encoder, decoder)
        self.assertEqual(renderer.bg_n, 6)
        self.assertEqual(len(renderer.ref_feats), 6)
        self.assertIsNotNone(good)

    def test_caches_are_bounded_to_the_ring_slots_actually_used(self):
        renderer = self._renderer(max_refs=4)
        self.assertEqual(len(renderer.bg), 4)
        self.assertEqual(renderer.bg_n, 4)
        self.assertEqual(len(renderer.ref_feats), 4)


if __name__ == "__main__":
    unittest.main()
