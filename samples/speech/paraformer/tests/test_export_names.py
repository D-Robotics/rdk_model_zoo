"""Physical output names must not collide with upstream internal tensor names."""

import unittest
import importlib.util


@unittest.skipUnless(importlib.util.find_spec("onnx"), "ONNX required")
class ExportOutputNames(unittest.TestCase):
    def test_internal_collision_preserves_dependency_chain(self):
        import onnx
        from onnx import helper as h, TensorProto as T
        from samples.speech.paraformer.conversion.export import bind_output_names
        from samples.speech.paraformer.runtime.python.model_binding import CONTEXT

        graph = h.make_model(
            h.make_graph(
                [
                    h.make_node("Identity", ["speech"], [CONTEXT]),
                    h.make_node("Identity", [CONTEXT], [CONTEXT + ".1"]),
                ],
                "collision",
                [h.make_tensor_value_info("speech", T.FLOAT, [1, 400, 512])],
                [h.make_tensor_value_info(CONTEXT + ".1", T.FLOAT, [1, 400, 512])],
            ),
            opset_imports=[h.make_opsetid("", 15)],
            ir_version=10,
        )
        result = bind_output_names(graph, "encoder")
        onnx.checker.check_model(result)
        self.assertEqual(result.graph.output[0].name, CONTEXT)
        self.assertNotEqual(result.graph.node[0].output[0], CONTEXT)
        self.assertEqual(result.graph.node[0].output[0], result.graph.node[1].input[0])
        self.assertEqual(result.graph.node[1].output[0], CONTEXT)
