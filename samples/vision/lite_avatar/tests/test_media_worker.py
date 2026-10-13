"""media_worker contracts: color-space gold vectors, sidecar, hardware decode.

The limited-range BT.601 integer matrices are pinned by gold vectors
computed from the standard fixed-point formulas (the same convention the
previous OpenCV ``COLOR_BGR2YUV_I420`` pipeline used).  The background
loader is driven against a fake ``hobot_vio`` module injected into
``sys.modules``: the real SDK exists only on boards, and the loader's
contract (empty-buffer bound, exact sidecar-count stop, close-always) is
board-independent.  No OpenCV is imported here at all.
"""

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from samples.vision.lite_avatar.runtime.python import media_worker


# ======================================================================
# Gold vectors: limited-range BT.601 (standard integer coefficients).
# ======================================================================

# (BGR color, Y, U, V) — U/V are the 2x2-subsampled chroma for solids.
SOLID_GOLD = [
    ((0, 0, 0), 16, 128, 128),  # black: limited-range floor
    ((255, 255, 255), 235, 128, 128),  # white: limited-range ceiling
    ((128, 128, 128), 126, 128, 128),  # mid gray
    ((0, 0, 255), 82, 90, 240),  # red (BGR layout)
    ((0, 255, 0), 144, 54, 34),  # green: 129*255+16*256+128 = 145*256-1,
    #  rounds to 144 (one LSB below the ideal 145 — pinned intentionally,
    #  the measured OpenCV value is 145; the runtime keeps exact-integer
    #  arithmetic over per-pixel rounding exceptions)
    ((255, 0, 0), 41, 240, 110),  # blue (BGR layout)
]

# Inverse gold: (Y, U, V, expected BGR pixel).
INVERSE_GOLD = [
    (16, 128, 128, (0, 0, 0)),
    (235, 128, 128, (255, 255, 255)),
    (82, 90, 240, (0, 1, 255)),  # BGR red: green carries the 1-LSB residue
    (144, 54, 34, (0, 254, 0)),  # BGR green
    (41, 240, 110, (255, 0, 0)),  # BGR blue
]


def nv12_solid(y, u, v):
    return bytes([y, y, y, y, u, v])


class ColorConversionTests(unittest.TestCase):
    def test_bgr_to_nv12_gold_vectors(self):
        for bgr, y, u, v in SOLID_GOLD:
            with self.subTest(bgr=bgr):
                frame = np.full((2, 2, 3), bgr, np.uint8)
                self.assertEqual(
                    media_worker.bgr_to_nv12(frame), nv12_solid(y, u, v)
                )

    def test_nv12_to_bgr_gold_vectors(self):
        for y, u, v, expected in INVERSE_GOLD:
            with self.subTest(yuv=(y, u, v)):
                frame = media_worker.nv12_to_bgr(nv12_solid(y, u, v), 2, 2, 2)
                np.testing.assert_array_equal(frame[0, 0], expected)
                np.testing.assert_array_equal(frame[1, 1], expected)

    def test_conversions_reject_invalid_geometry(self):
        with self.assertRaises(ValueError):
            media_worker.bgr_to_nv12(np.zeros((3, 2, 3), np.uint8))  # odd h
        with self.assertRaises(ValueError):
            media_worker.bgr_to_nv12(np.zeros((2, 2, 4), np.uint8))  # not BGR
        with self.assertRaises(ValueError):
            media_worker.nv12_to_bgr(b"\x00" * 5, 2, 2, 2)  # short buffer
        with self.assertRaises(ValueError):
            media_worker.nv12_to_bgr(b"\x00" * 6, 3, 4, 2)  # odd width
        with self.assertRaises(ValueError):
            media_worker.nv12_to_bgr(b"\x00" * 6, 4, 2, 2)  # width > stride

    def test_nv12_to_bgr_honors_stride_padding(self):
        # Luma/UV rows are stride-padded; the visible width reads the first
        # ``width`` columns of each row, not the tight layout.
        y = np.full((2, 4), 200, np.uint8)
        uv = np.full((1, 4), 128, np.uint8)
        buf = np.concatenate([y.reshape(-1), uv.reshape(-1)]).tobytes()
        frame = media_worker.nv12_to_bgr(buf, width=2, stride=4, height=2)
        self.assertEqual(frame.shape, (2, 2, 3))
        # Y=200 alone (neutral chroma) -> 298*(200-16) >> 8 = 214 both axes.
        np.testing.assert_array_equal(frame, np.full((2, 2, 3), 214, np.uint8))

    def test_pad_right_edge_replicates_and_short_circuits(self):
        frame = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(3, 4, 3)
        padded = media_worker.pad_right(frame, 8)
        self.assertEqual(padded.shape, (3, 8, 3))
        np.testing.assert_array_equal(
            padded[:, 4:], np.repeat(frame[:, -1:], 4, axis=1)
        )
        self.assertIs(media_worker.pad_right(frame, 4), frame)
        self.assertIs(media_worker.pad_right(frame, 2), frame)

    def test_drain_encoder_packets_is_bounded(self):
        class FakeEncoder:
            def __init__(self, packets):
                self.packets = list(packets)

            def get_img(self):
                return self.packets.pop(0) if self.packets else None

        encoder = FakeEncoder([b"a", b"b", b"c"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "late.h264"
            with out.open("wb") as handle:
                media_worker.drain_encoder_packets(encoder, handle, tries=3)
            self.assertEqual(out.read_bytes(), b"abc")
        self.assertIsNone(encoder.get_img())  # drained to None within bound


# ======================================================================
# Fake board SDK: Decoder double injected as hobot_vio.libsrcampy.
# ======================================================================


class FakeDecoder:
    """Cycles ``script`` through get_img; records calls; tracks close."""

    last = None
    decode_ret = [0]
    script = ()
    delay = 0.0
    #: Close-failure knobs: an exception to raise, or a nonzero status to
    #: return (0/None mean success, mirroring the SDK convention).
    close_raises = None
    close_returns = 0

    def __init__(self):
        self.calls = []
        self.get_img_calls = 0
        self.closed = False
        self._pos = 0
        FakeDecoder.last = self

    def decode(self, path, chn, typ, width, height):
        self.calls.append((path, chn, typ, width, height))
        return type(self).decode_ret

    def get_img(self):
        import time

        if type(self).delay:
            time.sleep(type(self).delay)
        self.get_img_calls += 1
        payload = type(self).script[self._pos % len(type(self).script)]
        self._pos += 1
        return payload

    def close(self):
        self.closed = True
        if type(self).close_raises is not None:
            raise type(self).close_raises
        return type(self).close_returns


class FakeEncoder:
    """Queues one packet per encode_file; get_img pops; records close."""

    last = None

    def __init__(self):
        self.pending = []
        self.encode_calls = []
        self.closed = False
        FakeEncoder.last = self

    def encode(self, *args):
        self.encode_calls.append(args)
        return 0

    def encode_file(self, nv12):
        self.pending.append(bytes(nv12))
        return 0

    def get_img(self):
        return self.pending.pop(0) if self.pending else None

    def close(self):
        self.closed = True
        return 0


def install_fake_vio(testcase):
    """Patch sys.modules with a hobot_vio exposing the fakes."""

    module = types.ModuleType("hobot_vio")
    srcampy = types.ModuleType("hobot_vio.libsrcampy")
    srcampy.Decoder = FakeDecoder
    srcampy.Encoder = FakeEncoder
    module.libsrcampy = srcampy
    patcher = mock.patch.dict(
        sys.modules, {"hobot_vio": module, "hobot_vio.libsrcampy": srcampy}
    )
    patcher.start()
    testcase.addCleanup(patcher.stop)


def nv12_strided(frame_bgr, stride):
    """Tight bgr_to_nv12 output re-laid out with luma/UV rows at ``stride``."""

    h, w = frame_bgr.shape[:2]
    tight = np.frombuffer(media_worker.bgr_to_nv12(frame_bgr), np.uint8)
    y = tight[: h * w].reshape(h, w)
    uv = tight[h * w :].reshape(h // 2, w)
    padded = np.zeros((h * 3 // 2, stride), np.uint8)
    padded[:h, :w] = y
    padded[h:, :w] = uv
    return padded.reshape(-1).tobytes()


def write_prepared(
    data_dir,
    payloads,
    *,
    width,
    coded_width,
    height,
    frames=None,
    codec="h264",
    sha="correct",
):
    """Write bg_video.h264 + bg_video.json describing the given payloads."""

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    stream = b"\x00\x00\x00\x01fake-es" + b"".join(payloads)
    (data_dir / "bg_video.h264").write_bytes(stream)
    sidecar = {
        "codec": codec,
        "width": width,
        "coded_width": coded_width,
        "height": height,
        "fps": 30,
        "frames": len(payloads) if frames is None else frames,
    }
    if sha == "correct":
        sidecar["sha256"] = hashlib.sha256(stream).hexdigest()
    elif sha is not None:
        sidecar["sha256"] = sha
    (data_dir / "bg_video.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return data_dir


def background_payloads(count, width, coded_width, height):
    rng = np.random.default_rng(5)
    payloads, frames = [], []
    for index in range(count):
        frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        payload = nv12_strided(frame, coded_width)
        payloads.append(payload)
        frames.append(
            media_worker.nv12_to_bgr(payload, width, coded_width, height)
        )
    return payloads, frames


#: Decode-test geometry: visible 30 within a 32-byte stride (the sample's
#: 890-within-896 layout, scaled down), even height.
GEOM = {"width": 30, "coded_width": 32, "height": 8}


class BackgroundSidecarTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.payloads, self.frames = background_payloads(3, **GEOM)

    def test_valid_sidecar_reports_verified_geometry(self):
        data = write_prepared(self.root / "ok", self.payloads, **GEOM)
        spec = media_worker.read_background_sidecar(data)
        self.assertEqual(
            (spec["width"], spec["coded_width"], spec["height"], spec["frames"]),
            (30, 32, 8, 3),
        )
        self.assertEqual(spec["stream"], data / "bg_video.h264")

    def test_stale_stream_fails_the_sha_check(self):
        data = write_prepared(self.root / "stale", self.payloads, **GEOM)
        (data / "bg_video.h264").write_bytes(b"truncated")
        with self.assertRaises(media_worker.BackgroundFormatError) as raised:
            media_worker.read_background_sidecar(data)
        self.assertIn("SHA-256", str(raised.exception))

    def test_sidecar_digest_is_mandatory_and_wellformed(self):
        # Without a mandatory digest, a replaced or partially copied stream
        # could decode to a wrong count on targets whose decode() reports
        # none — the digest is the cross-check, so it may not be optional.
        cases = (
            ("missing", None),
            ("empty", ""),
            ("not hex", "z" * 64),
            ("too short", "a" * 63),
            ("too long", "a" * 65),
        )
        for label, sha in cases:
            with self.subTest(case=label):
                data = write_prepared(self.root / label, self.payloads, sha=sha, **GEOM)
                with self.assertRaises(media_worker.BackgroundFormatError) as raised:
                    media_worker.read_background_sidecar(data)
                self.assertIn("SHA-256", str(raised.exception))

    def test_eight_but_not_sixteen_aligned_coded_width_is_rejected(self):
        # The decoder's NV12 luma stride is align-16: a stream coded at a
        # mere 8-pixel multiple (e.g. 904 for a 898-wide source) decodes to
        # a tight buffer with a truncated UV plane.  The sidecar is the
        # contract that must refuse it.
        data = write_prepared(
            self.root / "align8", self.payloads,
            width=30, coded_width=40, height=8,
        )
        with self.assertRaises(media_worker.BackgroundFormatError) as raised:
            media_worker.read_background_sidecar(data)
        self.assertIn("16-pixel multiple", str(raised.exception))

    def test_valid_digest_in_upper_case_is_accepted(self):
        data = write_prepared(self.root / "upper", self.payloads, **GEOM)
        sidecar_path = data / "bg_video.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["sha256"] = sidecar["sha256"].upper()
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        spec = media_worker.read_background_sidecar(data)
        self.assertEqual(spec["frames"], 3)

    def test_legacy_mpeg4_directs_to_the_host_prep_command(self):
        data = write_prepared(self.root / "legacy", self.payloads, **GEOM)
        (data / "bg_video.h264").unlink()
        (data / "bg_video.json").unlink()
        (data / "bg_video.mp4").write_bytes(b"legacy")
        with self.assertRaises(media_worker.BackgroundFormatError) as raised:
            media_worker.read_background_sidecar(data)
        message = str(raised.exception)
        self.assertIn("bg_video.mp4", message)
        self.assertIn(media_worker.HOST_PREP_COMMAND, message)
        self.assertIn("never overwriting the original", message)

    def test_missing_asset_and_bad_sidecars_fail_loudly(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(media_worker.BackgroundFormatError):
            media_worker.read_background_sidecar(empty)
        missing = write_prepared(self.root / "missing", self.payloads, **GEOM)
        (missing / "bg_video.json").write_text("{}")
        with self.assertRaises(media_worker.BackgroundFormatError):
            media_worker.read_background_sidecar(missing)
        bad_codec = write_prepared(
            self.root / "badcodec", self.payloads, codec="mpeg4", **GEOM
        )
        with self.assertRaises(media_worker.BackgroundFormatError):
            media_worker.read_background_sidecar(bad_codec)
        odd = write_prepared(
            self.root / "odd", self.payloads, width=31, coded_width=32, height=8
        )
        with self.assertRaises(media_worker.BackgroundFormatError):
            media_worker.read_background_sidecar(odd)
        unaligned = write_prepared(
            self.root / "unaligned",
            self.payloads,
            width=30,
            coded_width=30,
            height=8,
        )
        with self.assertRaises(media_worker.BackgroundFormatError):
            media_worker.read_background_sidecar(unaligned)
        broken = write_prepared(self.root / "broken", self.payloads, **GEOM)
        (broken / "bg_video.json").write_text("{not json")
        with self.assertRaises(media_worker.BackgroundFormatError):
            media_worker.read_background_sidecar(broken)


class BackgroundDecodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.payloads, self.frames = background_payloads(3, **GEOM)
        install_fake_vio(self)
        # Reset the double's class-level knobs: earlier tests may have set
        # failure modes that must not leak into the next case.
        FakeDecoder.last = None
        FakeDecoder.decode_ret = [0]
        FakeDecoder.script = ()
        FakeDecoder.delay = 0.0
        FakeDecoder.close_raises = None
        FakeDecoder.close_returns = 0

    def _data(self, **kwargs):
        merged = {**GEOM, **kwargs}
        return write_prepared(self.root / "data", self.payloads, **merged)

    def test_s100_style_count_with_leading_empty_buffer(self):
        # decode() reports the frame count (S100/S100P shape [0, N]); the
        # feeder emits one empty buffer right after open, then real frames.
        FakeDecoder.decode_ret = [0, 3]
        FakeDecoder.script = (b"",) + tuple(self.payloads) + (self.payloads[0],)
        frames = media_worker.decode_background(self._data())
        self.assertEqual(len(frames), 3)
        for got, want in zip(frames, self.frames):
            np.testing.assert_array_equal(got, want)
        decoder = FakeDecoder.last
        self.assertEqual(
            decoder.calls, [(str(self.root / "data" / "bg_video.h264"), 0, 1, 32, 8)]
        )
        # Exactly the sidecar count: the feeder's looped wrap frame (the 5th
        # payload) is never read.
        self.assertEqual(decoder.get_img_calls, 4)
        self.assertTrue(decoder.closed)

    def test_s600_style_countless_return_stops_at_sidecar_count(self):
        # S600's decode() returns [0] with no count; the verified sidecar is
        # the authority for the frame count.
        FakeDecoder.decode_ret = [0]
        FakeDecoder.script = tuple(self.payloads) + (self.payloads[0],)
        frames = media_worker.decode_background(self._data())
        self.assertEqual(len(frames), 3)
        self.assertEqual(FakeDecoder.last.get_img_calls, 3)

    def test_max_frames_bounds_the_ring(self):
        FakeDecoder.decode_ret = [0, 3]
        FakeDecoder.script = tuple(self.payloads) * 2
        frames = media_worker.decode_background(self._data(), max_frames=2)
        self.assertEqual(len(frames), 2)
        np.testing.assert_array_equal(frames[1], self.frames[1])

    def test_sdk_count_mismatch_with_sidecar_is_an_error(self):
        FakeDecoder.decode_ret = [0, 9]
        FakeDecoder.script = tuple(self.payloads)
        with self.assertRaises(media_worker.BackgroundFormatError) as raised:
            media_worker.decode_background(self._data())
        self.assertIn("9 frames", str(raised.exception))
        self.assertTrue(FakeDecoder.last.closed)

    def test_open_failure_is_reported_and_closes(self):
        FakeDecoder.decode_ret = [1]
        FakeDecoder.script = tuple(self.payloads)
        with self.assertRaises(media_worker.BackgroundFormatError):
            media_worker.decode_background(self._data())
        self.assertTrue(FakeDecoder.last.closed)

    def test_close_failure_after_a_successful_decode_propagates(self):
        # Resources must actually be released: a decoder that decodes every
        # frame but fails to close is a failed decode, not a success.
        for label, knob in (
            ("nonzero status", {"close_returns": 1}),
            ("raised exception", {"close_raises": RuntimeError("vio teardown")}),
        ):
            with self.subTest(case=label):
                FakeDecoder.decode_ret = [0, 3]
                FakeDecoder.script = (b"",) + tuple(self.payloads)
                for key, value in knob.items():
                    setattr(FakeDecoder, key, value)
                with self.assertRaises(RuntimeError) as raised:
                    media_worker.decode_background(self._data())
                self.assertIn("close failed", str(raised.exception))
                self.assertTrue(FakeDecoder.last.closed)
                for key in knob:
                    setattr(FakeDecoder, key, 0 if key == "close_returns" else None)

    def test_primary_decode_error_wins_over_a_close_failure(self):
        # Cleanup noise must never mask the real failure.
        FakeDecoder.decode_ret = [0, 9]  # SDK/sidecar count mismatch
        FakeDecoder.script = tuple(self.payloads)
        FakeDecoder.close_raises = RuntimeError("close exploded too")
        with self.assertRaises(media_worker.BackgroundFormatError) as raised:
            media_worker.decode_background(self._data())
        self.assertIn("9 frames", str(raised.exception))
        self.assertNotIn("close", str(raised.exception))
        self.assertTrue(FakeDecoder.last.closed)

    def test_short_frame_buffer_is_rejected(self):
        FakeDecoder.decode_ret = [0, 3]
        FakeDecoder.script = (self.payloads[0], b"\x01\x02")
        with self.assertRaises(media_worker.BackgroundFormatError) as raised:
            media_worker.decode_background(self._data())
        self.assertIn("bytes", str(raised.exception))
        self.assertTrue(FakeDecoder.last.closed)

    def test_none_payload_and_empty_flood_are_bounded(self):
        FakeDecoder.decode_ret = [0, 3]
        FakeDecoder.script = (None,)
        with self.assertRaises(media_worker.BackgroundFormatError):
            media_worker.decode_background(self._data())
        self.assertTrue(FakeDecoder.last.closed)

        FakeDecoder.decode_ret = [0, 3]
        FakeDecoder.script = (b"",)
        with self.assertRaises(media_worker.BackgroundFormatError):
            media_worker.decode_background(self._data())
        self.assertTrue(FakeDecoder.last.closed)

    def test_stalled_decode_times_out(self):
        FakeDecoder.decode_ret = [0, 3]
        FakeDecoder.script = (self.payloads[0],)
        FakeDecoder.delay = 0.2
        with mock.patch.object(media_worker, "CALL_TIMEOUT_S", 0.01):
            with self.assertRaises(TimeoutError):
                media_worker.decode_background(self._data())
        self.assertTrue(FakeDecoder.last.closed)

    def test_legacy_asset_needs_no_sdk_import(self):
        # The sidecar error must surface even where hobot_vio cannot be
        # imported at all (the guidance precedes the SDK on the host).
        data = self._data()
        (data / "bg_video.h264").unlink()
        (data / "bg_video.json").unlink()
        (data / "bg_video.mp4").write_bytes(b"legacy")
        with mock.patch.dict(
            sys.modules, {"hobot_vio": None, "hobot_vio.libsrcampy": None}
        ):
            with self.assertRaises(media_worker.BackgroundFormatError):
                media_worker.decode_background(data)


class ParentVanishesAfter(io.StringIO):
    """stdout double whose reader disappears once N writes succeeded."""

    def __init__(self, writes_survive):
        super().__init__()
        self.survived = writes_survive

    def write(self, text):
        if self.survived <= 0:
            raise BrokenPipeError("reader went away")
        self.survived -= 1
        return super().write(text)


class WorkerMainTests(unittest.TestCase):
    """worker_main's own cleanup contracts, in-process against fakes.

    The ring/ACK protocol runs for real (mmap ring, real NV12 conversion,
    fake SDK encoder); the remux is patched out except where asserted.
    """

    SLOTS, HEIGHT, WIDTH = 2, 8, 8

    def setUp(self):
        install_fake_vio(self)
        FakeEncoder.last = None
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ring = self.root / "ring.bin"
        frame = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
        self.ring.write_bytes(frame.tobytes() * self.SLOTS)
        self.output = self.root / "out.mp4"

    def _run(self, stdin_text, stdout=None):
        real_stdin, real_stdout, real_argv = sys.stdin, sys.stdout, sys.argv
        sys.stdin = io.StringIO(stdin_text)
        sys.stdout = stdout if stdout is not None else io.StringIO()
        sys.argv = [
            "media_worker",
            "--ring", str(self.ring),
            "--width", str(self.WIDTH),
            "--height", str(self.HEIGHT),
            "--slots", str(self.SLOTS),
            "--output", str(self.output),
            "--audio", str(self.root / "a.wav"),
        ]
        try:
            media_worker.worker_main()
        finally:
            captured = sys.stdout
            sys.stdin, sys.stdout, sys.argv = real_stdin, real_stdout, real_argv
        return captured

    def test_end_path_unlinks_ring_and_remux_scratch(self):
        with mock.patch.object(media_worker.subprocess, "run") as remux:
            with mock.patch.object(media_worker, "verify_remuxed_tracks"):
                out = self._run("0\nEND\n")
        text = out.getvalue()
        self.assertIn("READY", text)
        self.assertIn("ACK 0", text)
        self.assertIn("DONE", text)
        self.assertFalse(self.ring.exists())
        self.assertFalse(Path(f"{self.output}.h264").exists())
        (command,) = remux.call_args.args
        self.assertIn("copy", command)  # video is never re-encoded
        self.assertTrue(FakeEncoder.last.closed)

    def test_done_is_printed_only_after_the_product_is_verified(self):
        # FFmpeg exit 0 proved the MP4 exists, not that it carries both
        # tracks (the silent S600 delivery): DONE must stay behind the
        # verification, and a failed verification leaves the artifacts for
        # diagnostics.
        out = io.StringIO()
        with mock.patch.object(
            media_worker.subprocess, "run",
            return_value=mock.Mock(returncode=0),
        ):
            with mock.patch.object(
                media_worker,
                "verify_remuxed_tracks",
                side_effect=RuntimeError(
                    "delivered MP4 holds no AAC audio packets: {'video': ...}"
                ),
            ):
                with self.assertRaises(SystemExit) as raised:
                    self._run("0\nEND\n", stdout=out)
        self.assertIn("no AAC audio packets", str(raised.exception))
        self.assertNotIn("DONE", out.getvalue())
        self.assertFalse(self.ring.exists())
        self.assertTrue(Path(f"{self.output}.h264").exists())

    def test_dead_parent_mid_protocol_cleans_ring_and_partial_stream(self):
        # The reader of stdout disappears between READY and the first ACK:
        # the ring must still be unlinked (the parent cannot), the partial
        # elementary stream is garbage (no remux will consume it) and the
        # encoder is closed before exiting.
        Path(f"{self.output}.h264").write_bytes(b"partial scratch")
        with self.assertRaises(SystemExit) as raised:
            self._run("0\nEND\n", stdout=ParentVanishesAfter(1))
        self.assertIn("parent went away", str(raised.exception))
        self.assertFalse(self.ring.exists())
        self.assertFalse(Path(f"{self.output}.h264").exists())
        self.assertTrue(FakeEncoder.last.closed)

    def test_dead_parent_after_end_never_remuxes(self):
        # The protocol finished (READY, ACK) but the ENCODED announcement
        # finds the parent gone: cleanup still runs and ffmpeg is never
        # launched for an output nobody asked for.
        with mock.patch.object(media_worker.subprocess, "run") as remux:
            with self.assertRaises(SystemExit):
                self._run("0\nEND\n", stdout=ParentVanishesAfter(2))
        remux.assert_not_called()
        self.assertFalse(self.ring.exists())
        self.assertFalse(Path(f"{self.output}.h264").exists())
        self.assertTrue(FakeEncoder.last.closed)

    def test_protocol_error_with_live_parent_keeps_the_stream(self):
        # A genuine worker failure with the parent still reading is
        # diagnostics for the parent's error report: the elementary stream
        # survives, but the ring is still released (the parent's own unlink
        # tolerates the file being gone).
        with self.assertRaises(ValueError) as raised:
            self._run("5\n")  # slot outside the ring
        self.assertIn("Invalid frame slot", str(raised.exception))
        self.assertFalse(self.ring.exists())
        self.assertTrue(Path(f"{self.output}.h264").exists())
        self.assertTrue(FakeEncoder.last.closed)


def lavfi_fixtures(root):
    """One second of H.264 elementary video + a 1.03 s 16 kHz mono WAV."""

    elementary = root / "video.h264"
    wav = root / "a.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=30",
            "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-x264-params", "bframes=0", str(elementary),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi",
            "-i", "sine=frequency=440:sample_rate=16000:duration=1.03",
            "-ac", "1", "-c:a", "pcm_s16le", str(wav),
        ],
        check=True,
    )
    return elementary, wav


class RemuxCommandTests(unittest.TestCase):
    """The worker's remux must deliver both tracks, on every FFmpeg.

    FFmpeg >= 6 (measured: 6.1.1 on S600, 9.0.2 on a workstation) exits 0
    from the old ``-shortest`` mux with the audio stream dropped entirely;
    4.4 cut it at the video end.  The construction is pinned without
    ``-shortest`` (the frame count already matches the audio length to
    within one frame period), and wherever a real FFmpeg exists the command
    runs against generated fixtures to prove an H.264+AAC result.
    """

    def test_command_copies_video_encodes_audio_and_never_trims(self):
        command = media_worker.remux_command("v.h264", "a.wav", "out.mp4", pad=6)
        self.assertNotIn("-shortest", command)
        self.assertIn("copy", command)
        self.assertIn("aac", command)
        self.assertIn("v.h264", command)
        self.assertIn("a.wav", command)
        self.assertIn("out.mp4", command)
        self.assertIn("crop_right=6", " ".join(command))
        self.assertEqual(
            media_worker.remux_command("v.h264", "a.wav", "out.mp4").count("-i"), 2
        )

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not on PATH")
    def test_real_remux_delivers_both_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            elementary, wav = lavfi_fixtures(root)
            output = root / "out.mp4"
            subprocess.run(
                media_worker.remux_command(elementary, wav, output),
                check=True,
                timeout=60,
            )
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-count_frames",
                    "-show_entries",
                    "stream=codec_type,codec_name,sample_rate,channels,"
                    "duration,nb_read_frames",
                    "-of", "json", str(output),
                ],
                check=True,
                capture_output=True,
            )
            streams = json.loads(probe.stdout)["streams"]
            by_type = {stream["codec_type"]: stream for stream in streams}
            self.assertEqual(set(by_type), {"video", "audio"})
            self.assertEqual(by_type["video"]["codec_name"], "h264")
            # ffprobe emits numbers as strings on some versions and ints on
            # others; the values, not the JSON types, are the contract.
            self.assertEqual(int(by_type["video"]["nb_read_frames"]), 30)
            audio = by_type["audio"]
            self.assertEqual(audio["codec_name"], "aac")
            self.assertEqual(int(audio["sample_rate"]), 16000)
            self.assertEqual(int(audio["channels"]), 1)
            # The audio is muxed whole (the wav is 1.03 s for the 1.0 s
            # video, mirroring the runtime's frame-count derivation); a
            # dropped or over-trimmed track fails here.
            self.assertGreater(float(audio["duration"]), 0.9)


class RemuxGuardTests(unittest.TestCase):
    """verify_remuxed_tracks: the DONE gate looks at the actual product."""

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not on PATH")
    def test_both_track_mp4_passes_with_packet_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            elementary, wav = lavfi_fixtures(root)
            output = root / "out.mp4"
            subprocess.run(
                media_worker.remux_command(elementary, wav, output),
                check=True,
                timeout=60,
            )
            packets = media_worker.verify_remuxed_tracks(output)
            self.assertEqual(packets["video"], ("h264", 30))
            self.assertEqual(packets["audio"][0], "aac")
            self.assertGreaterEqual(packets["audio"][1], 1)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not on PATH")
    def test_silent_audio_less_mp4_is_rejected(self):
        # Exactly what FFmpeg >= 6 delivered from the old -shortest mux:
        # exit 0, video fine, no audio stream at all.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "silent.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=30",
                    "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(output),
                ],
                check=True,
            )
            with self.assertRaises(RuntimeError) as raised:
                media_worker.verify_remuxed_tracks(output)
            self.assertIn("no AAC audio packets", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
