"""Canonical stage names and the thin-entry structure for the SigLIP sample.

The readable-runtime design requires the model file to expose
``preprocess``/``infer``/``postprocess``/``predict`` as the primary stage
API with ``pre_process``/``forward``/``post_process`` kept as compatibility
delegates, and the entry to parse arguments through a local ``cli`` module.
"""
import unittest

import numpy as np

from test_siglip import FACTS, FakeRuntime, meta


def task(variant='base-patch16-224', sub='pooler_output', dtype='F32'):
    from samples.vision.siglip.runtime.python.cli import resolve_selection
    from samples.vision.siglip.runtime.python.embedding import SigLIPEmbedder, bind_model

    class StubRunner:
        def __init__(self, binding, raw):
            self.binding, self.raw = binding, raw
        def load(self):
            return self.binding
        def __call__(self, tensors):
            return self.raw

    runtime = FakeRuntime(variant, dtype)
    binding = bind_model(resolve_selection('s100', variant=variant, submodel=sub), meta(variant, sub, dtype))
    return SigLIPEmbedder(resolve_selection('s100', variant=variant, submodel=sub), runner=StubRunner(binding, runtime.raw[sub])), runtime.raw[sub]


class CanonicalStageTests(unittest.TestCase):
    def test_canonical_stage_names_exist_and_delegate(self):
        for sub in ('pooler_output', 'last_hidden_state'):
            model, raw = task(sub=sub)
            image = np.zeros((32, 81, 3), dtype=np.uint8)

            prepared = model.preprocess(image)
            legacy = model.pre_process(image)
            np.testing.assert_array_equal(prepared.tensors['_input_0'], legacy.tensors['_input_0'])
            self.assertEqual(prepared.context, legacy.context)

            self.assertIs(model.infer(prepared.tensors), raw)
            result = model.postprocess(raw)
            np.testing.assert_array_equal(result, raw['_output_0'])
            self.assertFalse(np.shares_memory(result, raw['_output_0']))

            composed = model.predict(image)
            np.testing.assert_array_equal(
                composed, model.postprocess(model.infer(model.preprocess(image).tensors)))

    def test_predict_routes_through_canonical_stages(self):
        model, _ = task()
        calls = {'pre': 0, 'inf': 0, 'post': 0}
        original = (model.preprocess, model.infer, model.postprocess)

        def counting_preprocess(image):
            calls['pre'] += 1
            return original[0](image)

        def counting_infer(tensors):
            calls['inf'] += 1
            return original[1](tensors)

        def counting_postprocess(outputs):
            calls['post'] += 1
            return original[2](outputs)

        model.preprocess = counting_preprocess
        model.infer = counting_infer
        model.postprocess = counting_postprocess
        model.predict(np.zeros((10, 11, 3), np.uint8))
        self.assertEqual(calls, {'pre': 1, 'inf': 1, 'post': 1})

    def test_all_variants_keep_canonical_api(self):
        for variant, (size, _, _) in FACTS.items():
            model, _ = task(variant)
            prepared = model.preprocess(np.zeros((7, 9, 3), np.uint8))
            self.assertEqual(prepared.tensors['_input_0'].shape, (1, 3, size, size))


class ThinEntryTests(unittest.TestCase):
    def test_main_reexports_local_cli_parser(self):
        from samples.vision.siglip.runtime.python import cli, main
        self.assertIs(main.build_parser, cli.build_parser)
        for name in ('run_list_models', 'run_dry_run', 'print_summary'):
            self.assertTrue(hasattr(cli, name), name)


if __name__ == '__main__':
    unittest.main()
