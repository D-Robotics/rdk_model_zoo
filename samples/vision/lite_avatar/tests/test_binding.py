"""Published identity, tensor binding and four-runner composition (SDK doubles)."""

from dataclasses import replace
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from samples.vision.lite_avatar.runtime.python import avatar, cli
from samples.vision.lite_avatar.runtime.python.frontend import AvatarFrontend
from utils.py_utils.runtime_meta import RuntimeMetadata
from samples.vision.lite_avatar.tests.doubles import FakeRuntime

SAMPLE_DIR = Path(avatar.__file__).resolve().parents[2]
CMVN = SAMPLE_DIR / "model" / "am.mvn"

STAGE_BY_FILENAME = {
    Path(cli.FILENAMES[stage]).name: stage for stage in cli.STAGES
}


def factory_for_stages(recorder=None):
    """A runtime factory building one FakeRuntime per published filename.

    ``recorder`` collects an ordered event trace of what actually reaches
    the runtimes: ``("run", stage)`` for every inference and
    ``("sched", stage)`` for every scheduling request, in arrival order.
    """

    def factory(path):
        runtime = FakeRuntime(STAGE_BY_FILENAME[Path(path).name])
        if recorder is not None:
            original_run = runtime.run

            def traced_run(feed):
                recorder.append(("run", runtime.stage))
                return original_run(feed)

            runtime.run = traced_run
            runtime.set_scheduling_params = lambda **kwargs: recorder.append(
                ("sched", runtime.stage)
            )
        return runtime

    return factory


class SelectionTests(unittest.TestCase):
    def test_only_the_published_s100_four_model_group_is_selected(self):
        selections = cli.resolve_selections("s100")
        self.assertEqual(
            tuple(selection.stage for selection in selections),
            ("audio_encoder", "audio2mouth", "facegen_encoder", "facegen_decoder"),
        )
        self.assertTrue(
            all(s.asset.reference.startswith("s:lite_avatar:s100/") for s in selections)
        )
        self.assertTrue(all(s.target == "s100" for s in selections))

    def test_unpublished_targets_are_rejected(self):
        for target in ("s100p", "s600", "x5", "invalid", ""):
            with self.subTest(target=target), self.assertRaises(ValueError):
                cli.resolve_selections(target)

    def test_every_manifest_digest_is_a_real_sha256(self):
        for selection in cli.resolve_selections("s100"):
            digest = selection.asset.sha256
            self.assertIsInstance(digest, str)
            self.assertEqual(len(digest), 64)
            self.assertTrue(all(char in "0123456789abcdef" for char in digest))

    def test_local_hash_manifest_identifies_explicit_rebuilds(self):
        golden = cli.resolve_selections("s100")
        mapping = {s.asset.filename: "0" * 64 for s in golden}
        path = Path(self.id().replace(".", "_") + ".json")
        self.addCleanup(path.unlink)
        for records in (
            {"s100/lite_avatar_encoder_400x560_s100.hbm": "ZZ"},
            {"s100/unknown.hbm": "0" * 64},
            ["not", "an", "object"],
            {},
        ):
            path.write_text(json.dumps(records), encoding="utf-8")
            with self.subTest(records=records), self.assertRaises(ValueError):
                cli.load_local_hash_manifest(path)
        path.write_text(json.dumps(mapping), encoding="utf-8")
        loaded = cli.load_local_hash_manifest(path)
        self.assertEqual(loaded, {name: "0" * 64 for name in mapping})
        rebuilt = cli.resolve_selections("s100", local_hashes=loaded)
        self.assertTrue(all(s.asset.sha256 == "0" * 64 for s in rebuilt))
        self.assertNotEqual(rebuilt[0].asset, golden[0].asset)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.selections = cli.resolve_selections("s100")

    def test_bind_resolves_roles_and_accepts_board_dtype_tokens(self):
        for selection in self.selections:
            bound = avatar.bind_model(
                selection, RuntimeMetadata.from_runtime(FakeRuntime(selection.stage))
            )
            self.assertEqual(bound.model_name, selection.stage)
            self.assertEqual(set(bound.inputs), set(avatar.INPUTS[selection.stage]))
            self.assertEqual(set(bound.outputs), set(avatar.OUTPUTS[selection.stage]))

    def test_binding_is_order_independent(self):
        selection = self.selections[3]
        runtime = FakeRuntime(selection.stage)
        runtime.output_names = {
            selection.stage: list(reversed(runtime.output_names[selection.stage]))
        }
        bound = avatar.bind_model(
            selection, RuntimeMetadata.from_runtime(runtime)
        )
        self.assertEqual(bound.inputs["mouth_param"], "mouth_param")

    def test_rejects_wrong_dtype_shape_extra_names_or_extra_models(self):
        selection = self.selections[0]
        cases = [("int16 boundary", FakeRuntime(selection.stage, dtype_token="int16"))]
        runtime = FakeRuntime(selection.stage)
        shapes = dict(runtime.input_shapes[selection.stage])
        shapes["feats"] = (1, 400, 512)
        runtime.input_shapes = {selection.stage: shapes}
        cases.append(("wrong input shape", runtime))
        runtime = FakeRuntime(selection.stage)
        names = dict(runtime.output_names)
        names[selection.stage] = names[selection.stage] + ["extra"]
        runtime.output_names = names
        cases.append(("extra output", runtime))
        runtime = FakeRuntime(selection.stage)
        runtime.model_names = [selection.stage, "second_model"]
        cases.append(("two models", runtime))
        for label, runtime in cases:
            with self.subTest(case=label), self.assertRaises(ValueError):
                avatar.bind_model(selection, RuntimeMetadata.from_runtime(runtime))

    def test_selection_cannot_relabel_stage_or_undeclared_path(self):
        selection = self.selections[0]
        for changed in (
            replace(selection, stage="facegen_decoder"),
            replace(selection, model_path=Path("/tmp/undeclared.hbm")),
        ):
            with self.assertRaises(ValueError):
                avatar.validate_selection(changed)


class RunnerCompositionTests(unittest.TestCase):
    def _load(self):
        selections = cli.resolve_selections("s100")
        return (
            avatar.load_model_runners(
                selections, runtime_factory=factory_for_stages()
            ),
            selections,
        )

    def _pipeline(self):
        (runners, names, quants), _ = self._load()
        pipeline = avatar.LiteAvatarPipeline(
            AvatarFrontend(CMVN), runners, names, "/tmp", quants
        )
        return pipeline, runners, names

    def test_loads_binds_then_releases_facegen_pair(self):
        (runners, names, quants), _ = self._load()
        self.assertEqual(
            tuple(quant is not None for quant in quants), (True, True, True, True)
        )
        self.assertEqual(names.audio_encoder_input, "feats")
        self.assertEqual(names.audio2mouth_feat, "feat")
        self.assertEqual(names.facegen_decoder_mouth, "mouth_param")
        self.assertEqual(names.facegen_decoder_output, "mouth_image")
        # All four bindings were validated, then the facegen pair was unloaded
        # (memory discipline); only the audio pair stays resident.
        self.assertTrue(runners[0].loaded and runners[1].loaded)
        self.assertFalse(runners[2].loaded or runners[3].loaded)

    def test_rejects_wrong_stage_order_or_non_s100_group(self):
        selections = list(cli.resolve_selections("s100"))
        reordered = [selections[1], selections[0], selections[2], selections[3]]
        with self.assertRaises(ValueError):
            avatar.load_model_runners(reordered, runtime_factory=factory_for_stages())
        foreign = [replace(selection, target="s600") for selection in selections]
        with self.assertRaises(ValueError):
            avatar.load_model_runners(foreign, runtime_factory=factory_for_stages())

    def test_pipeline_composes_stages_and_releases_audio_models(self):
        pipeline, runners, names = self._pipeline()
        self.assertEqual(pipeline.audio_encoder_stage.stage, "audio_encoder")
        self.assertEqual(pipeline.facegen_decoder_stage.stage, "facegen_decoder")
        self.assertFalse(pipeline.audio_released)
        pipeline.release_audio_models()
        self.assertTrue(pipeline.audio_released)
        self.assertFalse(runners[0].loaded or runners[1].loaded)
        # Idempotent: a second release must not raise.
        pipeline.release_audio_models()

    def test_scheduling_precedes_the_first_run_of_every_load(self):
        events = []
        selections = cli.resolve_selections("s100")
        runners, names, quants = avatar.load_model_runners(
            selections, runtime_factory=factory_for_stages(recorder=events)
        )
        pipeline = avatar.LiteAvatarPipeline(
            AvatarFrontend(CMVN), runners, names, "/tmp", quants
        )
        pipeline.set_scheduling_params(priority=8, bpu_cores=[0, 1])
        # The two audio runtimes are the loaded ones at construction, so the
        # request reaches them before any inference happens at all.
        self.assertEqual(
            events,
            [("sched", "audio_encoder"), ("sched", "audio2mouth")],
        )

        # Second-predict path: after the audio release, the lazy reload
        # applies the stored request BEFORE the fresh runtime's first run.
        pipeline.release_audio_models()
        feed = pipeline.audio_encoder_stage.preprocess(
            np.zeros((3, 560), np.float32)
        ).tensors
        pipeline.audio_encoder_stage.infer(feed)
        self.assertEqual(
            events[2:],
            [("sched", "audio_encoder"), ("run", "audio_encoder")],
        )

        # The first face-model load (renderer-init handoff) follows the same
        # order: scheduling first, then the very first inference.
        prepared = pipeline.facegen_encoder_stage.preprocess(
            np.zeros((1, 3, 384, 384), np.float32)
        )
        pipeline.facegen_encoder_stage.infer(prepared.tensors)
        self.assertEqual(
            events[4:],
            [("sched", "facegen_encoder"), ("run", "facegen_encoder")],
        )

    def test_partial_scheduling_updates_keep_earlier_values(self):
        (runners, names, quants), _ = self._load()
        pipeline = avatar.LiteAvatarPipeline(
            AvatarFrontend(CMVN), runners, names, "/tmp", quants
        )
        pipeline.set_scheduling_params(priority=8)
        pipeline.set_scheduling_params(bpu_cores=[1])
        # The second, partial call must not discard the stored priority.
        pipeline.release_audio_models()
        feed = pipeline.audio_encoder_stage.preprocess(
            np.zeros((3, 560), np.float32)
        ).tensors
        seen = {}
        runner = runners[0]
        original = runner.set_scheduling_params

        def trace(**kwargs):
            seen.update(kwargs)
            original(**kwargs)

        runner.set_scheduling_params = trace
        pipeline.audio_encoder_stage.infer(feed)
        self.assertEqual(seen, {"priority": 8, "bpu_cores": [1]})
        for bad in (
            {"priority": 300},
            {"priority": -1},
            {"bpu_cores": []},
            {"bpu_cores": [0, -2]},
        ):
            with self.subTest(request=bad), self.assertRaises(ValueError):
                pipeline.set_scheduling_params(**bad)

    def test_local_rebuild_selection_binds_and_loads(self):
        golden = cli.resolve_selections("s100")
        mapping = {s.asset.filename: "1" * 64 for s in golden}
        selections = cli.resolve_selections("s100", local_hashes=mapping)
        self.assertTrue(all(s.digest_source == "local" for s in selections))
        self.assertTrue(all(s.asset.sha256 == "1" * 64 for s in selections))
        # The full bind + load route accepts explicitly identified rebuilds.
        for selection in selections:
            bound = avatar.bind_model(
                selection, RuntimeMetadata.from_runtime(FakeRuntime(selection.stage))
            )
            self.assertEqual(bound.model_name, selection.stage)
        runners, names, quants = avatar.load_model_runners(
            selections, runtime_factory=factory_for_stages()
        )
        self.assertTrue(runners[0].loaded and runners[1].loaded)

        # A local digest may not smuggle a different asset identity, and an
        # undeclared digest change stays rejected.
        with self.assertRaises(ValueError):
            avatar.validate_selection(
                replace(
                    selections[0],
                    asset=replace(selections[0].asset, filename="s100/other.hbm"),
                )
            )
        with self.assertRaises(ValueError):
            avatar.validate_selection(
                replace(
                    golden[0], asset=replace(golden[0].asset, sha256="1" * 64)
                )
            )

    def test_direct_api_local_hashes_are_validated(self):
        with self.assertRaises(ValueError):
            cli.resolve_selections(
                "s100",
                local_hashes={"s100/lite_avatar_encoder_400x560_s100.hbm": "zz"},
            )


if __name__ == "__main__":
    unittest.main()
