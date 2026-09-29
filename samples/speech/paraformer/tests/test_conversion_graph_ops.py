"""Actual ONNX/ORT checks for graph rewriting; not model or OE validation."""

import unittest
from unittest.mock import patch
import numpy as np

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = ort = None


@unittest.skipIf(onnx is None or ort is None, "ONNX conversion dependencies required")
class GraphOperations(unittest.TestCase):
    def setUp(self):
        from samples.speech.paraformer.conversion import graph_ops

        self.ops = graph_ops
        self.h = onnx.helper
        self.nh = onnx.numpy_helper
        self.tp = onnx.TensorProto

    def info(self, name, dtype, shape):
        return self.h.make_tensor_value_info(name, dtype, shape)

    def model(self, nodes, inputs, outputs, initializers=()):
        result = self.h.make_model(
            self.h.make_graph(
                nodes, "fixture", inputs, outputs, initializer=list(initializers)
            ),
            opset_imports=[self.h.make_opsetid("", 15)],
            ir_version=10,
        )
        result.producer_name = "preserve-me"
        result.doc_string = "explicit host graph fixture"
        return result

    def equal(self, first, second, feeds):
        onnx.checker.check_model(second)
        a = ort.InferenceSession(
            first.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, feeds)
        b = ort.InferenceSession(
            second.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, feeds)
        for x, y in zip(a, b, strict=True):
            self.assertEqual(x.dtype, y.dtype)
            np.testing.assert_array_equal(x, y)

    def test_toposort_preserves_nodes_with_duplicate_names_and_metadata(self):
        h = self.h
        model = self.model(
            [
                h.make_node("Identity", ["mid"], ["out"], name="same"),
                h.make_node("Identity", ["x"], ["mid"], name="same"),
            ],
            [self.info("x", self.tp.FLOAT, [1])],
            [self.info("out", self.tp.FLOAT, [1])],
        )
        before = model.SerializeToString()
        result = self.ops.topological_sort(model)
        self.assertEqual([n.output[0] for n in result.graph.node], ["mid", "out"])
        self.assertEqual(result.producer_name, "preserve-me")
        self.assertEqual(model.SerializeToString(), before)
        onnx.checker.check_model(result)

    def test_toposort_rejects_cycle_missing_and_duplicate_producers(self):
        for nodes in (
            [
                self.h.make_node("Identity", ["b"], ["a"]),
                self.h.make_node("Identity", ["a"], ["b"]),
            ],
            [self.h.make_node("Identity", ["missing"], ["a"])],
            [
                self.h.make_node("Identity", ["x"], ["a"]),
                self.h.make_node("Identity", ["x"], ["a"]),
            ],
        ):
            model = self.model(
                nodes,
                [self.info("x", self.tp.FLOAT, [1])],
                [self.info("a", self.tp.FLOAT, [1])],
            )
            with self.assertRaises(ValueError):
                self.ops.topological_sort(model)

    def test_gather_does_not_mutate_shared_int64_constant(self):
        model = self.model(
            [
                self.h.make_node("Gather", ["x", "indices"], ["g"]),
                self.h.make_node("Add", ["indices", "offset"], ["other"]),
            ],
            [self.info("x", self.tp.FLOAT, [2])],
            [
                self.info("g", self.tp.FLOAT, [1]),
                self.info("other", self.tp.INT64, [1]),
            ],
            [
                self.nh.from_array(np.array([1], np.int64), "indices"),
                self.nh.from_array(np.array([2], np.int64), "offset"),
            ],
        )
        before = model.SerializeToString()
        result = self.ops.gather_indices_int32(model)
        self.equal(model, result, {"x": np.array([3, 9], np.float32)})
        self.assertEqual(
            next(i for i in result.graph.initializer if i.name == "indices").data_type,
            self.tp.INT64,
        )
        self.assertEqual(model.SerializeToString(), before)

    def test_dynamic_gather_is_explicit_and_overflow_rejected(self):
        model = self.model(
            [self.h.make_node("Gather", ["x", "index"], ["out"])],
            [
                self.info("x", self.tp.FLOAT, [2]),
                self.info("index", self.tp.INT64, [1]),
            ],
            [self.info("out", self.tp.FLOAT, [1])],
        )
        with self.assertRaises(ValueError):
            self.ops.gather_indices_int32(model)
        result = self.ops.gather_indices_int32(model, allow_dynamic=True)
        for index in (-1, 0, 1):
            self.equal(
                model,
                result,
                {
                    "x": np.array([3, 9], np.float32),
                    "index": np.array([index], np.int64),
                },
            )
        constant = self.model(
            list(model.graph.node),
            [model.graph.input[0]],
            list(model.graph.output),
            [self.nh.from_array(np.array([2**40], np.int64), "index")],
        )
        with self.assertRaises(ValueError):
            self.ops.gather_indices_int32(constant)

    def test_constant_over_budget_is_not_treated_as_dynamic(self):
        model = self.model(
            [self.h.make_node("Gather", ["x", "index"], ["out"])],
            [self.info("x", self.tp.FLOAT, [2])],
            [self.info("out", self.tp.FLOAT, [2])],
            [self.nh.from_array(np.array([0, 2**40], np.int64), "index")],
        )
        with patch.object(self.ops, "MAX_CONSTANT_ELEMENTS", 1):
            with self.assertRaisesRegex(ValueError, "Constant Gather index"):
                self.ops.gather_indices_int32(model, allow_dynamic=True)

    def test_multiple_dynamic_casts_have_distinct_tensor_names(self):
        model = self.model(
            [
                self.h.make_node("Gather", ["x", "i"], ["a"]),
                self.h.make_node("Gather", ["x", "j"], ["b"]),
            ],
            [
                self.info("x", self.tp.FLOAT, [2]),
                self.info("i", self.tp.INT64, [1]),
                self.info("j", self.tp.INT64, [1]),
            ],
            [self.info("a", self.tp.FLOAT, [1]), self.info("b", self.tp.FLOAT, [1])],
        )
        result = self.ops.gather_indices_int32(model, allow_dynamic=True)
        self.equal(
            model,
            result,
            {
                "x": np.array([3, 9], np.float32),
                "i": np.array([0], np.int64),
                "j": np.array([1], np.int64),
            },
        )
        names = [name for node in result.graph.node for name in node.output]
        self.assertEqual(len(names), len(set(names)))

    def test_shape_derived_range_folds_without_data_probe(self):
        nodes = [
            self.h.make_node("Shape", ["x"], ["shape"]),
            self.h.make_node("Gather", ["shape", "zero"], ["limit"]),
            self.h.make_node("Range", ["zero", "limit", "one"], ["range"]),
        ]
        model = self.model(
            nodes,
            [self.info("x", self.tp.FLOAT, [4])],
            [self.info("range", self.tp.INT64, [4])],
            [
                self.nh.from_array(np.array(0, np.int64), "zero"),
                self.nh.from_array(np.array(1, np.int64), "one"),
            ],
        )
        result = self.ops.fold_constant_ranges(model)
        self.assertNotIn("Range", [n.op_type for n in result.graph.node])
        for value in (0, 17):
            self.equal(model, result, {"x": np.full(4, value, np.float32)})

    def test_range_from_constant_maximum_is_proven_without_probe(self):
        model = self.model(
            [
                self.h.make_node("ReduceMax", ["lengths"], ["limit"], keepdims=0),
                self.h.make_node("Range", ["zero", "limit", "one"], ["range"]),
            ],
            [],
            [self.info("range", self.tp.INT64, [4])],
            [
                self.nh.from_array(np.array([2, 4], np.int64), "lengths"),
                self.nh.from_array(np.array(0, np.int64), "zero"),
                self.nh.from_array(np.array(1, np.int64), "one"),
            ],
        )
        result = self.ops.fold_constant_ranges(model)
        self.assertNotIn("Range", [n.op_type for n in result.graph.node])
        self.equal(model, result, {})

    def test_dynamic_range_cannot_be_frozen_from_one_example(self):
        model = self.model(
            [self.h.make_node("Range", ["zero", "limit", "one"], ["range"])],
            [self.info("limit", self.tp.INT64, [])],
            [self.info("range", self.tp.INT64, [None])],
            [
                self.nh.from_array(np.array(0, np.int64), "zero"),
                self.nh.from_array(np.array(1, np.int64), "one"),
            ],
        )
        before = model.SerializeToString()
        with self.assertRaises(ValueError):
            self.ops.fold_constant_ranges(model)
        self.assertEqual(model.SerializeToString(), before)

    def test_shared_axes_are_copied_per_consumer_rank(self):
        nodes = [
            self.h.make_node("Squeeze", ["x", "axes"], ["sx"]),
            self.h.make_node("Squeeze", ["y", "axes"], ["sy"]),
            self.h.make_node("Identity", ["axes"], ["original_axes"]),
        ]
        model = self.model(
            nodes,
            [
                self.info("x", self.tp.FLOAT, [1, 2, 1]),
                self.info("y", self.tp.FLOAT, [1, 1]),
            ],
            [
                self.info("sx", self.tp.FLOAT, [1, 2]),
                self.info("sy", self.tp.FLOAT, [1]),
                self.info("original_axes", self.tp.INT64, [1]),
            ],
            [self.nh.from_array(np.array([-1], np.int64), "axes")],
        )
        result = self.ops.normalize_axes(model)
        self.equal(
            model,
            result,
            {"x": np.array([[[1], [2]]], np.float32), "y": np.array([[3]], np.float32)},
        )
        axes = {i.name: self.nh.to_array(i).tolist() for i in result.graph.initializer}
        self.assertEqual(axes["axes"], [-1])
        self.assertEqual(axes[result.graph.node[0].input[1]], [2])
        self.assertEqual(axes[result.graph.node[1].input[1]], [1])


if __name__ == "__main__":
    unittest.main()
