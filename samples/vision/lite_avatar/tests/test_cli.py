"""CLI surface: parser, validation, encoder resolution, listing, delivery."""

import io
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from samples.vision.lite_avatar.runtime.python import cli


def namespace(**overrides):
    values = dict(
        target="s100",
        list_models=False,
        dry_run=False,
        audio=Path("in.wav"),
        avatar_data=Path("data"),
        cmvn_path=Path("am.mvn"),
        output=Path("out.mp4"),
        encoder="auto",
        no_encode=False,
        model_dir=None,
        models_manifest=None,
        priority=None,
        bpu_cores=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class ParserTests(unittest.TestCase):
    def test_defaults_match_the_documented_surface(self):
        args = cli.build_parser().parse_args([])
        self.assertEqual(args.target, "auto")
        self.assertIsNone(args.audio)
        self.assertEqual(args.encoder, "auto")
        self.assertFalse(args.no_encode)
        self.assertEqual(args.output, Path("outputs/lite_avatar/avatar.mp4"))
        self.assertEqual(args.avatar_data, cli.SAMPLE_DIR / "test_data" / "preload")
        self.assertEqual(args.cmvn_path, cli.SAMPLE_DIR / "model" / "am.mvn")

    def test_rejects_unknown_flags(self):
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["--nonsense"])


class NormalizeArgsTests(unittest.TestCase):
    def test_rejects_bad_flag_combinations_and_values(self):
        cases = (
            ("priority", namespace(priority=300)),
            ("negative priority", namespace(priority=-1)),
            ("empty bpu cores", namespace(bpu_cores=[])),
            ("negative bpu core", namespace(bpu_cores=[0, -2])),
            (
                "no-encode with encoder",
                namespace(no_encode=True, encoder="software"),
            ),
            (
                "hardware encoder on x5",
                namespace(target="x5", encoder="hardware"),
            ),
        )
        for label, args in cases:
            with self.subTest(case=label), self.assertRaises(ValueError):
                cli.normalize_args(args)

    def test_real_runs_require_an_existing_audio_file(self):
        with self.assertRaises(ValueError):
            cli.normalize_args(namespace(audio=None))
        with self.assertRaises(ValueError):
            cli.normalize_args(namespace(audio=Path("/nonexistent/x.wav")))

    def test_paths_are_expanded_in_place(self):
        real = Path(__file__).resolve()
        args = namespace(audio=real)
        cli.normalize_args(args)
        self.assertEqual(args.audio, real.resolve())


class EncoderResolutionTests(unittest.TestCase):
    def test_auto_resolves_to_the_verified_encoder_only_on_s100(self):
        self.assertEqual(cli.resolve_encoder("auto", "s100"), "hardware")
        self.assertEqual(cli.resolve_encoder("auto", "x5"), "software")
        self.assertEqual(cli.resolve_encoder("auto", "s600"), "software")

    def test_explicit_choices_are_never_silently_changed(self):
        self.assertEqual(cli.resolve_encoder("software", "x5"), "software")
        self.assertEqual(cli.resolve_encoder("software", "s100"), "software")
        self.assertEqual(cli.resolve_encoder("hardware", "s100"), "hardware")
        with self.assertRaises(ValueError):
            cli.resolve_encoder("hardware", "s600")
        with self.assertRaises(ValueError):
            cli.resolve_encoder("ffmpeg", "s100")


class ModelListingTests(unittest.TestCase):
    def test_listing_names_every_published_reference_without_an_sdk(self):
        buffer = io.StringIO()
        code = 0
        with redirect_stdout(buffer):
            code = cli.print_model_list("s100")
        text = buffer.getvalue()
        self.assertEqual(code, 0)
        for filename in cli.FILENAMES.values():
            self.assertIn(filename, text)
        self.assertIn("s:lite_avatar:s100/", text)
        self.assertIn("prepare locally", text)

    def test_resolution_reports_missing_artifacts_with_a_failing_status(self):
        selections = cli.resolve_selections("s100")
        missing = cli.missing_models(selections)
        # The host test tree ships only am.mvn: every artifact is missing.
        self.assertEqual(len(missing), 4)
        args = namespace(list_models=False, dry_run=True)
        buffer, errbuf = io.StringIO(), io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(errbuf):
            code = cli.print_resolution(args, selections)
        self.assertEqual(code, 1)
        self.assertIn("model_path_exists: False", buffer.getvalue())
        self.assertIn("Missing artifacts", buffer.getvalue())
        self.assertIn("No model is downloaded and no SDK is loaded", buffer.getvalue())

    def test_dry_run_auto_requires_the_explicit_target(self):
        args = namespace(target="auto", dry_run=True)
        with self.assertRaises(ValueError):
            cli.resolve_selections_for(args)


class DeliverFramesTests(unittest.TestCase):
    class FakeResult:
        def __init__(self, count, fail_at=None):
            self.count = count
            self.fail_at = fail_at
            self.rendered = []

        @property
        def frame_count(self):
            return self.count

        def render_frame(self, index):
            if self.fail_at is not None and index == self.fail_at:
                raise RuntimeError("render exploded")
            self.rendered.append(index)
            return f"frame-{index}"

    class FakeWriter:
        def __init__(self):
            self.submitted = []
            self.closed = False
            self.aborted = False

        def submit(self, frame):
            self.submitted.append(frame)

        def close(self):
            self.closed = True

        def abort(self):
            self.aborted = True

    def test_renders_every_frame_and_closes_the_writer(self):
        result, writer = self.FakeResult(4), self.FakeWriter()
        seconds = cli.deliver_frames(result, writer)
        self.assertGreaterEqual(seconds, 0.0)
        self.assertEqual(result.rendered, [0, 1, 2, 3])
        self.assertEqual(writer.submitted, [f"frame-{i}" for i in range(4)])
        self.assertTrue(writer.closed)
        self.assertFalse(writer.aborted)

    def test_without_a_writer_frames_still_render(self):
        result = self.FakeResult(3)
        cli.deliver_frames(result, None)
        self.assertEqual(result.rendered, [0, 1, 2])

    def test_render_failure_aborts_the_writer_and_propagates(self):
        result, writer = self.FakeResult(4, fail_at=2), self.FakeWriter()
        with self.assertRaises(RuntimeError):
            cli.deliver_frames(result, writer)
        self.assertFalse(writer.closed)
        self.assertTrue(writer.aborted)


class ReadAudioTests(unittest.TestCase):
    def test_reads_a_wav_when_soundfile_is_available(self):
        try:
            import soundfile  # noqa: F401
        except ImportError:
            self.skipTest("soundfile not installed on this host")
        import tempfile

        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.wav"
            data = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
            import soundfile as sf

            # FLOAT subtype: the round trip is exact for float32 samples, so
            # the 1e-6 gate below tests the loader, not PCM_16 quantization.
            sf.write(str(path), data, 16000, subtype="FLOAT")
            waveform, rate = cli.read_audio(path)
            self.assertEqual(rate, 16000)
            self.assertEqual(waveform.dtype, np.float32)
            np.testing.assert_allclose(waveform, data, atol=1e-6)
            # The default WAV subtype quantizes to PCM_16; the loader must
            # still return that dequantized signal unchanged in float32
            # (truncation, so up to one full step of 1/32768).
            sf.write(str(path), data, 16000)
            quantized, _ = cli.read_audio(path)
            self.assertEqual(quantized.dtype, np.float32)
            np.testing.assert_allclose(quantized, data, atol=1 / 32768)


if __name__ == "__main__":
    unittest.main()
