"""Per-stage contracts, the a2m window loop and the full predict flow.

Every model call goes through ``FakeRuntime`` doubles with the verified F32
boundary; the Torch-dependent steps (fbank, time interpolation) are replaced
by deterministic NumPy doubles whose real numerics are pinned on the server
(parity run) and the board (paired comparison).
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from samples.vision.lite_avatar.runtime.python import avatar, frontend
from samples.vision.lite_avatar.runtime.python.frontend import AvatarFrontend
from samples.vision.lite_avatar.tests.doubles import (
    ArrayFrontend,
    FakeRuntime,
    build_avatar_data,
    numpy_interpolate,
)

SAMPLE_DIR = Path(avatar.__file__).resolve().parents[2]
CMVN = SAMPLE_DIR / "model" / "am.mvn"


class AudioEncoderStageTests(unittest.TestCase):
    def _stage(self):
        return avatar.AudioEncoderStage(
            FakeRuntime("audio_encoder"), "feats", "hidden_states", None
        )

    def test_preprocess_pads_to_the_fixed_window(self):
        stage = self._stage()
        features = np.ones((7, 560), np.float32)
        prepared = stage.preprocess(features)
        window = prepared.tensors["feats"]
        self.assertEqual(window.shape, (1, 400, 560))
        self.assertTrue(np.array_equal(window[0, :7], features))
        self.assertTrue(np.all(window[0, 7:] == 0))
        self.assertEqual(prepared.context.feature_frames, 7)

    def test_preprocess_rejects_unusable_feature_matrices(self):
        stage = self._stage()
        for bad in (
            np.ones((401, 560), np.float32),  # beyond the encoder window
            np.ones((0, 560), np.float32),
            np.ones((3, 559), np.float32),
            np.ones((3, 560), np.float64),
            np.ones((3, 560, 1), np.float32),
        ):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                stage.preprocess(bad)
        poisoned = np.ones((3, 560), np.float32)
        poisoned[1, 0] = np.nan
        with self.assertRaises(ValueError):
            stage.preprocess(poisoned)

    def test_postprocess_trims_to_real_frames_and_copies(self):
        stage = self._stage()
        outputs = {
            "hidden_states": np.arange(400 * 50 * 512, dtype=np.float32).reshape(
                1, 400, 50, 512
            )
            / 1e6
        }
        hidden = stage.postprocess(outputs, avatar.AvatarContext(9, 0))
        self.assertEqual(hidden.shape, (9, 50, 512))
        self.assertTrue(
            np.array_equal(hidden, outputs["hidden_states"][0, :9])
        )
        hidden[:] = -1.0  # owned copy: the raw output stays untouched
        self.assertTrue((outputs["hidden_states"][0, :9, 0, 0] >= 0).all())

    def test_infer_attributes_failures_to_stage_and_operation(self):
        stage = self._stage()
        prepared = stage.preprocess(np.ones((2, 560), np.float32))
        with mock.patch.object(
            stage, "runner", side_effect=RuntimeError("sdk blew up")
        ), self.assertRaises(avatar.StageError) as raised:
            stage.infer(prepared.tensors)
        self.assertEqual(raised.exception.stage, "audio_encoder")
        self.assertIn("infer", str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)


class Audio2MouthWindowTests(unittest.TestCase):
    def _stage(self, transform=None):
        runner = FakeRuntime("audio2mouth", transform=transform)
        return avatar.Audio2MouthStage(runner, "input_au", "output", "feat", None)

    @staticmethod
    def _echo(feed):
        # Each output row echoes its window row's marker coefficient.
        marker = feed["input_au"][0, :, 0, 0]
        rows = np.repeat(marker[:, None], 32, axis=1)
        return {"output": rows[None], "feat": np.zeros((1, 30, 256), np.float32)}

    def test_window_loop_assembles_every_frame_exactly_once(self):
        stage = self._stage(self._echo)
        frames = 100
        au = np.zeros((frames, 50, 512), np.float32)
        au[:, 0, 0] = np.arange(frames, dtype=np.float32)
        with mock.patch.object(avatar, "_butter_lowpass", lambda data: data):
            params = stage.predict(au)
        self.assertEqual(params.shape, (frames, 32))
        # Identity mapping through the sliding windows and crossfades: any
        # mishandled overlap or pad would show up as a duplicated or missing
        # frame index (float32 blend rounding stays far below this bound).
        np.testing.assert_allclose(
            params, np.repeat(np.arange(frames)[:, None], 32, axis=1), atol=1e-3
        )

    def test_window_loop_pads_short_clips_with_the_last_parameters(self):
        stage = self._stage(self._echo)
        frames = 5
        au = np.zeros((frames, 50, 512), np.float32)
        au[:, 0, 0] = np.arange(frames, dtype=np.float32)
        with mock.patch.object(avatar, "_butter_lowpass", lambda data: data):
            params = stage.predict(au)
        self.assertEqual(params.shape, (frames, 32))
        np.testing.assert_allclose(
            params, np.repeat(np.arange(frames)[:, None], 32, axis=1), atol=1e-3
        )

    def test_preprocess_pads_windows_and_rejects_bad_input(self):
        stage = self._stage()
        short = stage.preprocess(np.ones((11, 50, 512), np.float32))
        self.assertEqual(short.tensors["input_au"].shape, (1, 30, 50, 512))
        self.assertTrue(np.all(short.tensors["input_au"][0, 11:] == 0))
        with self.assertRaises(ValueError):
            stage.preprocess(np.ones((31, 50, 512), np.float32))
        with self.assertRaises(ValueError):
            stage.preprocess(np.ones((30, 50, 256), np.float32))

    def test_butterworth_smoothing_pins_the_source_parameters(self):
        from scipy.signal import butter, filtfilt

        data = np.cumsum(np.random.default_rng(7).normal(size=(67, 32)), axis=0)
        data = data.astype(np.float32)
        expected = filtfilt(
            *butter(4, 2 * 10 / 30, "lowpass", analog=False),
            data,
            padtype=None,
            axis=0,
        ).astype(np.float32)
        self.assertTrue(np.array_equal(avatar._butter_lowpass(data), expected))
        # Low-pass character: a 14.5 Hz component (cutoff 10 Hz, Nyquist 15)
        # vanishes in the interior while a 1 Hz component survives.
        time = np.arange(90) / 30.0
        slow = np.sin(2 * np.pi * 1.0 * time)
        fast = 0.5 * np.sin(2 * np.pi * 14.5 * time)
        smoothed_mixed = avatar._butter_lowpass((slow + fast).astype(np.float32))
        smoothed_slow = avatar._butter_lowpass(slow.astype(np.float32))
        self.assertLess(
            np.abs(smoothed_mixed[15:-15] - smoothed_slow[15:-15]).max(), 1e-3
        )
        self.assertGreater(np.abs(smoothed_slow[15:-15]).max(), 0.9)


class InterpolateTests(unittest.TestCase):
    def test_rejects_bad_geometry_or_frame_counts(self):
        good = np.zeros((4, 50, 512), np.float32)
        for bad in (
            np.zeros((4, 50, 511), np.float32),
            np.zeros((4, 49, 512), np.float32),
            np.zeros(50),
            np.zeros((2, 3, 4, 5), np.float32),
        ):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                avatar.interpolate_time_axis(bad, 5)
        poisoned = np.zeros((4, 50, 512), np.float32)
        poisoned[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            avatar.interpolate_time_axis(poisoned, 5)
        for count in (0, -1, 100_001, 1.5, True, None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                avatar.interpolate_time_axis(good, count)

    # The Torch interpolation path (F.interpolate linear resample) is
    # verified on the board against the source implementation: the host
    # suite is deliberately Torch-free and only checks the rejecting path.


class FaceGenBorrowedOutputTests(unittest.TestCase):
    def test_pyramid_and_mouth_outputs_are_owned_copies(self):
        encoder = avatar.FaceGenEncoderStage(
            FakeRuntime("facegen_encoder"), "image", ("f0", "f1", "f2", "f3"), None
        )
        raw = {
            "f0": np.zeros((1, 3, 384, 384), np.float32),
            "f1": np.zeros((1, 16, 192, 192), np.float32),
            "f2": np.zeros((1, 32, 96, 96), np.float32),
            "f3": np.zeros((1, 64, 48, 48), np.float32),
        }
        pyramid = encoder.postprocess(raw)
        self.assertEqual(len(pyramid), 4)
        pyramid[0][:] = 9.0  # mutating the copy must not touch the raw buffer
        self.assertTrue(np.all(raw["f0"] == 0))

        decoder = avatar.FaceGenDecoderStage(
            FakeRuntime("facegen_decoder"),
            ("f0", "f1", "f2", "f3"),
            "mouth_param",
            "mouth_image",
            None,
        )
        raw = {
            "f0": np.zeros((1, 3, 384, 384), np.float32),
            "f1": np.zeros((1, 16, 192, 192), np.float32),
            "f2": np.zeros((1, 32, 96, 96), np.float32),
            "f3": np.zeros((1, 64, 48, 48), np.float32),
            "mouth_param": np.zeros((1, 32), np.float32),
        }
        prepared = decoder.preprocess(
            (
                raw["f0"],
                raw["f1"],
                raw["f2"],
                raw["f3"],
            ),
            np.full((1, 32), 0.5, np.float32),
        )
        feed = prepared.tensors
        self.assertEqual(set(feed), {"f0", "f1", "f2", "f3", "mouth_param"})
        # The decoder feed is owned: later mutation of the caller's pyramid
        # cannot change what was validated.
        self.assertTrue(np.all(feed["mouth_param"] == 0.5))
        mouth = decoder.postprocess(
            {"mouth_image": np.zeros((1, 3, 384, 384), np.float32)}
        )
        self.assertEqual(mouth.shape, (3, 384, 384))

    def test_preprocess_rejects_wrong_pyramid_arity(self):
        decoder = avatar.FaceGenDecoderStage(
            FakeRuntime("facegen_decoder"),
            ("f0", "f1", "f2", "f3"),
            "mouth_param",
            "mouth_image",
            None,
        )
        zeros = tuple(
            np.zeros(shape, np.float32)
            for shape in ((1, 3, 384, 384), (1, 16, 192, 192))
        )
        with self.assertRaises(ValueError):
            decoder.preprocess(zeros, np.zeros((1, 32), np.float32))


class QuantOwnershipTests(unittest.TestCase):
    """Stages own detached quant records; SDK descriptors are not retained."""

    class _FakeQuant:
        """Native-like SDK descriptor: attribute surface, refuses copying."""

        def __init__(self, quant_type, scale, zero_point, axis):
            self.quant_type = quant_type
            self.scale = scale
            self.zero_point = zero_point
            self.axis = axis

        def __deepcopy__(self, memo):
            raise TypeError("SDK descriptors refuse deepcopy (QuantParams)")

    def _descriptor(self):
        import types

        scale = np.array([0.5, 2.0], np.float32)
        zero_point = np.array([3, 4], np.int64)
        return self._FakeQuant(
            types.SimpleNamespace(name="SCALE"), scale, zero_point, 0
        )

    def test_stage_releases_the_sdk_descriptor_and_keeps_owned_copies(self):
        import gc
        import weakref

        descriptor = self._descriptor()
        scale_alias = descriptor.scale
        watch = weakref.ref(descriptor)
        stage = avatar.AudioEncoderStage(
            FakeRuntime("audio_encoder"),
            "feats",
            "hidden_states",
            {"hidden_states": descriptor},
        )
        del descriptor
        gc.collect()
        self.assertIsNone(
            watch(), "the stage still references the runtime-owned descriptor"
        )
        owned = stage.quants["hidden_states"]
        self.assertIsInstance(owned, avatar.OwnedQuantDescriptor)
        self.assertEqual(owned.quant_type, "SCALE")
        self.assertEqual(owned.axis, 0)
        np.testing.assert_array_equal(owned.scale, scale_alias)
        np.testing.assert_array_equal(owned.zero_point, [3, 4])
        # Owned copies: mutating the record cannot reach the SDK arrays.
        owned.scale[0] = 99.0
        self.assertEqual(scale_alias[0], 0.5)

    def test_detached_records_drive_the_shared_dequantization_chain(self):
        descriptor = self._descriptor()
        raw = {"mouth": np.array([10, 20], np.int32)}
        with_sdk = avatar.float_outputs(dict(raw), {"mouth": descriptor})
        owned = avatar.detach_quants({"mouth": descriptor})
        with_owned = avatar.float_outputs(dict(raw), owned)
        for value in (with_sdk["mouth"], with_owned["mouth"]):
            self.assertEqual(value.dtype, np.float32)
        np.testing.assert_array_equal(with_sdk["mouth"], with_owned["mouth"])
        self.assertTrue(np.isfinite(with_owned["mouth"]).all())

    def test_detach_is_idempotent_and_preserves_none_descriptors(self):
        owned = avatar.OwnedQuantDescriptor(
            "SCALE", np.float32(0.5), np.array(1, np.int64), None
        )
        detached = avatar.detach_quants({"a": None, "b": owned})
        self.assertIsNone(detached["a"])
        self.assertIs(detached["b"], owned)
        self.assertIs(avatar.detach_quants(None), None)

    def test_foreign_descriptor_shapes_fail_loudly(self):
        for bad in (
            {"scale": 0.5},  # plain mapping: not the SDK surface
            object(),  # no quant_type at all
        ):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError):
                    avatar.detach_quants({"mouth": bad})
        # A SCALE descriptor without scale/zero_point is unusable, not null.
        broken = self._descriptor()
        broken.scale = None
        with self.assertRaises(ValueError):
            avatar.detach_quants({"mouth": broken})

    def test_load_model_runners_snapshots_survive_the_facegen_unload(self):
        from samples.vision.lite_avatar.runtime.python import cli
        from samples.vision.lite_avatar.tests.test_binding import factory_for_stages

        selections = cli.resolve_selections("s100")
        _, _, quants = avatar.load_model_runners(
            selections, runtime_factory=factory_for_stages()
        )
        self.assertEqual(len(quants), 4)
        for stage_quants in quants:
            for name, descriptor in stage_quants.items():
                self.assertIsNone(
                    descriptor,
                    f"the F32/NONE double boundary must keep {name} empty",
                )
        # The facegen runners were unloaded inside load_model_runners; their
        # quant snapshots are plain owned mappings, not runtime references.
        self.assertTrue(all(isinstance(q, dict) for q in quants))


class PipelinePredictTests(unittest.TestCase):
    """Full predict flow twice: release, reload, renderer cache, ownership."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_dir = build_avatar_data(Path(self.tmp.name) / "avatar")
        self.encoder_calls = []

        def encoder_transform(feed):
            self.encoder_calls.append(1)
            return {
                name: np.zeros(shape, np.float32)
                for name, shape in FakeRuntime("facegen_encoder").output_shapes_l.items()
            }

        self.encoder_calls.clear()

        from samples.vision.lite_avatar.tests.test_binding import (
            factory_for_stages,
        )

        stage_transforms = {
            "facegen_encoder": encoder_transform,
        }

        def factory(path):
            from samples.vision.lite_avatar.tests.test_binding import (
                STAGE_BY_FILENAME,
            )

            stage = STAGE_BY_FILENAME[Path(path).name]
            return FakeRuntime(stage, transform=stage_transforms.get(stage))

        from samples.vision.lite_avatar.runtime.python import cli

        self.cli = cli
        selections = cli.resolve_selections("s100")
        self.runners, self.names, self.quants = avatar.load_model_runners(
            selections, runtime_factory=factory
        )
        self.pipeline = avatar.LiteAvatarPipeline(
            ArrayFrontend(CMVN), self.runners, self.names, self.data_dir, self.quants
        )

    def _predict(self):
        audio = np.zeros(24000, np.float32)  # 1.5 s -> 45 output frames
        with mock.patch.object(avatar, "interpolate_time_axis", numpy_interpolate):
            return self.pipeline.predict(avatar.AvatarInput(audio, 16000))

    def test_two_predictions_follow_the_declared_session_semantics(self):
        first = self._predict()
        self.assertEqual(first.frame_count, 45)
        self.assertEqual((first.width, first.height), (64, 48))
        self.assertEqual(first.params.shape, (45, 32))
        self.assertTrue(np.isfinite(first.params).all())
        # After predict: audio pair released, encoder loaded by the renderer's
        # reference pass, decoder still lazy (it loads per rendered frame).
        self.assertTrue(self.pipeline.audio_released)
        self.assertFalse(self.runners[0].loaded or self.runners[1].loaded)
        self.assertTrue(self.runners[2].loaded)
        self.assertFalse(self.runners[3].loaded)
        first_frame = first.render_frame(0)
        self.assertEqual(first_frame.shape, (48, 64, 3))
        self.assertTrue(self.runners[3].loaded)
        refs_encoded = len(self.encoder_calls)
        renderer = self.pipeline._renderer
        self.assertIsNotNone(renderer)
        self.assertEqual(len(renderer.ref_feats), renderer.bg_n)

        second = self._predict()
        self.assertEqual(second.frame_count, 45)
        # The renderer and its reference pyramids are reused, not rebuilt.
        self.assertIs(self.pipeline._renderer, renderer)
        self.assertEqual(len(self.encoder_calls), refs_encoded)
        # The audio models ran again and were released again.
        self.assertTrue(self.pipeline.audio_released)
        self.assertFalse(self.runners[0].loaded or self.runners[1].loaded)
        self.assertEqual(set(self.pipeline.last_timings), {"audio", "renderer_init"})

    def test_rendered_frames_are_owned_and_wrap_the_parameter_track(self):
        result = self._predict()
        frame = result.render_frame(0)
        self.assertEqual(frame.shape, (48, 64, 3))
        self.assertEqual(frame.dtype, np.uint8)
        frame[:] = 255  # mutating the delivered frame must not poison the cache
        again = result.render_frame(0)
        self.assertFalse(np.all(again == 255))
        with self.assertRaises(ValueError):
            result.render_frame(-1)
        with self.assertRaises(ValueError):
            result.render_frame(True)
        frames = list(result.frames())
        self.assertEqual(len(frames), 45)
        for a, b in zip(frames[:-1], frames[1:]):
            self.assertIsNot(a, b)

    def test_predict_rejects_unusable_audio_and_wrong_input_type(self):
        with self.assertRaises(ValueError):
            self.pipeline.predict("not an input")
        with self.assertRaises(ValueError):
            self.pipeline.predict(
                avatar.AvatarInput(np.zeros(100, np.float32), 16000)  # < 1 frame
            )
        # A failing audio stage keeps the release flag truthful.
        broken = ArrayFrontend(CMVN)
        broken.features_from_mono = lambda audio: (_ for _ in ()).throw(
            RuntimeError("fbank failed")
        )
        self.pipeline.frontend = broken
        with mock.patch.object(
            avatar, "interpolate_time_axis", numpy_interpolate
        ), self.assertRaises(RuntimeError):
            self.pipeline.predict(avatar.AvatarInput(np.zeros(24000, np.float32), 16000))
        self.assertFalse(self.pipeline.audio_released)


if __name__ == "__main__":
    unittest.main()
