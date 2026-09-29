"""Published model identity and real shared-runner composition with SDK doubles."""

from dataclasses import replace
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from samples._shared.runtime_meta import RuntimeMetadata


def metadata(stage):
    context = "/encoder/after_norm/Add_1_output_0"
    contracts = {
        "encoder": (
            {"speech": ((1, 400, 560), "float32")},
            {context: ((1, 400, 512), "float32")},
        ),
        "predictor": (
            {context: ((1, 400, 512), "float32")},
            {
                "/predictor/Add_output_0": ((1, 401), "float32"),
                "/predictor/Concat_5_output_0": ((1, 401, 512), "float32"),
            },
        ),
        "decoder": (
            {
                context: ((1, 400, 512), "float32"),
                "token_num": ((1,), "int32"),
                "bias_embed": ((1, 1, 512), "float32"),
                "onnx::Shape_8609": ((1, 100, 512), "float32"),
            },
            {"logits": ((1, 100, 8404), "float32"), "token_num": ((1,), "int32")},
        ),
    }
    inputs, outputs = contracts[stage]
    return RuntimeMetadata.from_mapping(
        {
            "model_name": stage,
            "input_names": tuple(inputs),
            "output_names": tuple(outputs),
            "input_shapes": {n: s for n, (s, _) in inputs.items()},
            "input_dtypes": {n: d for n, (_, d) in inputs.items()},
            "output_shapes": {n: s for n, (s, _) in outputs.items()},
            "output_dtypes": {n: d for n, (_, d) in outputs.items()},
        }
    )


class BindingTests(unittest.TestCase):
    def setUp(self):
        name = "samples.speech.paraformer.runtime.python.model_binding"
        self.assertIsNotNone(importlib.util.find_spec(name), "Missing model binding")
        self.binding = importlib.import_module(name)

    def test_only_exact_published_three_model_group_is_selected(self):
        selections = self.binding.resolve_selections("s100")
        self.assertEqual(
            tuple(s.stage for s in selections), ("encoder", "predictor", "decoder")
        )
        self.assertTrue(
            all(s.asset.reference.startswith("s:paraformer:s100/") for s in selections)
        )
        for target in ("s100p", "s600", "x5", "s", "invalid"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.binding.resolve_selections(target)

    def test_external_files_require_all_explicit_matching_identities(self):
        selected = self.binding.resolve_selections("s100")
        paths = {s.stage: f"/tmp/{s.stage}.hbm" for s in selected}
        ids = {s.stage: s.asset.reference for s in selected}
        with self.assertRaises(ValueError):
            self.binding.resolve_selections("s100", model_paths=paths)
        actual = self.binding.resolve_selections(
            "s100", model_paths=paths, asset_ids=ids
        )
        self.assertTrue(all(s.explicit_model_path for s in actual))
        self.assertEqual(actual[0].model_path, Path("/tmp/encoder.hbm"))
        ids["decoder"] = ids["encoder"]
        with self.assertRaises(ValueError):
            self.binding.resolve_selections("s100", model_paths=paths, asset_ids=ids)

    def test_named_binding_is_order_independent_and_accepts_count_output(self):
        for selection in self.binding.resolve_selections("s100"):
            meta = metadata(selection.stage)
            meta = replace(
                meta,
                input_names=tuple(reversed(meta.input_names)),
                output_names=tuple(reversed(meta.output_names)),
            )
            bound = self.binding.bind_model(selection, meta)
            self.assertEqual(bound.model_name, selection.stage)
            self.assertEqual(bound.metadata, meta)
        self.assertEqual(bound.outputs["logits"], "logits")

    def test_rejects_wrong_tensor_type_geometry_extra_or_unknown_names(self):
        selected = self.binding.resolve_selections("s100")[1]
        meta = metadata("predictor")
        out = meta.output_names[0]
        cases = [
            replace(meta, output_dtypes={**meta.output_dtypes, out: "int16"}),
            replace(meta, output_shapes={**meta.output_shapes, out: (1, 400)}),
            replace(meta, model_names=("predictor", "other")),
            replace(meta, output_names=meta.output_names + ("extra",)),
            replace(meta, output_names=("unknown", meta.output_names[1])),
        ]
        for changed in cases:
            with self.subTest(meta=changed), self.assertRaises(ValueError):
                self.binding.bind_model(selected, changed)

    def test_selection_cannot_relabel_stage_or_model_path(self):
        selection = self.binding.resolve_selections("s100")[0]
        for changed in (
            replace(selection, stage="decoder"),
            replace(selection, model_path=Path("/tmp/undeclared.hbm")),
        ):
            with self.assertRaises(ValueError):
                self.binding.bind_model(changed, metadata("encoder"))

    def test_invalid_group_is_rejected_before_any_sdk_factory_call(self):
        runtime_module = importlib.import_module(
            "samples.speech.paraformer.runtime.python.runtime"
        )
        selected = list(self.binding.resolve_selections("s100"))
        selected[2] = replace(selected[2], asset=selected[0].asset)
        calls = []

        def factory(path):
            calls.append(path)
            raise RuntimeError("SDK factory must not run")

        with self.assertRaises(ValueError):
            runtime_module.load_runtime(
                selected, [f"t{i}" for i in range(8404)], runtime_factory=factory
            )
        self.assertEqual(calls, [])

    def test_decoder_accepts_documented_acoustic_alias_and_no_passthrough(self):
        selection = self.binding.resolve_selections("s100")[2]
        meta = metadata("decoder")
        old, new = "onnx::Shape_8609", "shape_8609"
        meta = replace(
            meta,
            input_names=tuple(new if n == old else n for n in meta.input_names),
            input_shapes={
                new if n == old else n: v for n, v in meta.input_shapes.items()
            },
            input_dtypes={
                new if n == old else n: v for n, v in meta.input_dtypes.items()
            },
            output_names=("logits",),
            output_shapes={"logits": (1, 100, 8404)},
            output_dtypes={"logits": "float32"},
        )
        result = self.binding.bind_model(selection, meta)
        self.assertEqual(result.inputs["acoustic"], new)
        self.assertEqual(result.outputs, {"logits": "logits"})

    def test_shared_runners_execute_pipeline_and_delegate_all_scheduling(self):
        runtime_module = importlib.import_module(
            "samples.speech.paraformer.runtime.python.runtime"
        )
        runtimes = {}

        def factory(path):
            stage = next(s for s in ("encoder", "predictor", "decoder") if s in path)
            meta = metadata(stage)
            runtime = SimpleNamespace(model_names=[stage])
            for field in (
                "input_names",
                "input_shapes",
                "input_dtypes",
                "output_names",
                "output_shapes",
                "output_dtypes",
            ):
                setattr(runtime, field, {stage: getattr(meta, field)})
            runtime.scheduling = []
            runtime.set_scheduling_params = lambda **kwargs: runtime.scheduling.append(
                kwargs
            )
            runtime.calls = []

            def run(inputs):
                runtime.calls.append(inputs)
                outputs = {
                    name: np.zeros(shape, dtype=meta.output_dtypes[name])
                    for name, shape in meta.output_shapes.items()
                }
                if stage == "predictor":
                    outputs["/predictor/Add_output_0"][0, :2] = 1
                if stage == "decoder":
                    outputs["logits"][0, :2, 3] = 1
                    outputs["token_num"][:] = 2
                return {stage: outputs}

            runtime.run = run
            runtimes[stage] = runtime
            return runtime

        selected = self.binding.resolve_selections("s100")
        vocabulary = [f"token{i}" for i in range(8404)]
        bundle = runtime_module.load_runtime(
            selected, vocabulary, runtime_factory=factory
        )
        bundle.set_scheduling_params(priority=7, bpu_cores=[0])
        result = bundle.pipeline.predict(np.zeros((1, 400, 560), np.float32), 2)
        self.assertEqual(result.text, "token3token3")
        for stage, runtime in runtimes.items():
            self.assertEqual(
                runtime.scheduling,
                [{"priority": {stage: 7}, "bpu_cores": {stage: [0]}}],
            )
            self.assertEqual(len(runtime.calls), 1)
        runtimes["decoder"].set_scheduling_params = None
        with self.assertRaises(RuntimeError):
            bundle.set_scheduling_params(priority=8)
        self.assertEqual(len(runtimes["encoder"].scheduling), 1)


if __name__ == "__main__":
    unittest.main()
