"""Shared, non-mutating flat-ONNX rewrites for fixed-shape Paraformer export.

These are graph primitives, not a full exporter or proof of OE compatibility.
Never freeze data-dependent values from a single runtime observation.
"""

from collections import deque
import copy
import math

import numpy as np
import onnx
from onnx import helper, numpy_helper as nh, TensorProto, AttributeProto
from onnx.reference import ReferenceEvaluator

MAX_CONSTANT_ELEMENTS = 1_000_000


def topological_sort(model):
    """Clone and stable-sort by tensor dependency; reject incomplete graphs."""
    result = copy.deepcopy(model)
    graph = result.graph
    nodes = list(graph.node)
    base = {i.name for i in graph.input} | {i.name for i in graph.initializer}
    producer = {}
    for index, node in enumerate(nodes):
        if any(
            a.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS)
            for a in node.attribute
        ):
            raise ValueError(
                "Extract flat subgraphs before applying these graph operations"
            )
        for name in node.output:
            if not name:
                continue
            if name in producer or name in base:
                raise ValueError(f"Duplicate tensor producer: {name}")
            producer[name] = index
    dependencies = []
    for node in nodes:
        missing = [
            name
            for name in node.input
            if name and name not in base and name not in producer
        ]
        if missing:
            raise ValueError(f"Missing tensor producer: {missing}")
        dependencies.append({producer[name] for name in node.input if name in producer})
    followers = [[] for _ in nodes]
    degree = [len(d) for d in dependencies]
    for index, deps in enumerate(dependencies):
        for dependency in sorted(deps):
            followers[dependency].append(index)
    ready = deque(i for i, count in enumerate(degree) if not count)
    order = []
    while ready:
        index = ready.popleft()
        order.append(index)
        for follower in followers[index]:
            degree[follower] -= 1
            if degree[follower] == 0:
                ready.append(follower)
    if len(order) != len(nodes):
        raise ValueError("Cycle in ONNX graph; no partial graph will be returned")
    if any(o.name not in base and o.name not in producer for o in graph.output):
        raise ValueError("Missing graph output producer")
    del graph.node[:]
    graph.node.extend(nodes[i] for i in order)
    return result


def _metadata(model):
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except (onnx.shape_inference.InferenceError, onnx.checker.ValidationError):
        inferred = model
    return {
        v.name: v
        for v in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }


def _shape(value):
    if (
        value is None
        or not value.type.HasField("tensor_type")
        or not value.type.tensor_type.HasField("shape")
    ):
        return None
    dims = value.type.tensor_type.shape.dim
    return [int(d.dim_value) if d.HasField("dim_value") else None for d in dims]


def _constant_values(model):
    """Evaluate only whitelisted deterministic constant/known-shape paths."""
    metadata = _metadata(model)
    inputs = {i.name for i in model.graph.input}
    values = {
        i.name: nh.to_array(i)
        for i in model.graph.initializer
        if i.name not in inputs and math.prod(i.dims) <= MAX_CONSTANT_ELEMENTS
    }
    pure = {
        "Constant",
        "Identity",
        "Cast",
        "Gather",
        "Slice",
        "Concat",
        "Unsqueeze",
        "Squeeze",
        "Reshape",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Floor",
        "Ceil",
        "Range",
        "ReduceMax",
        "ReduceMin",
        "Transpose",
    }
    opsets = {o.domain: o.version for o in model.opset_import}
    for node in model.graph.node:
        if node.domain:
            continue
        if node.op_type == "Shape":
            shape = _shape(metadata.get(node.input[0]))
            if shape is not None and all(n is not None for n in shape):
                attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                values[node.output[0]] = np.asarray(
                    shape[attrs.get("start", 0) : attrs.get("end", len(shape))],
                    dtype=np.int64,
                )
            continue
        if node.op_type not in pure or any(
            name and name not in values for name in node.input
        ):
            continue
        if any(
            a.type == AttributeProto.TENSOR
            and math.prod(a.t.dims) > MAX_CONSTANT_ELEMENTS
            for a in node.attribute
        ):
            continue
        if any(
            (shape := _shape(metadata.get(name))) is not None
            and all(d is not None for d in shape)
            and math.prod(shape) > MAX_CONSTANT_ELEMENTS
            for name in node.output
        ):
            continue
        if node.op_type == "Range":
            scalars = [values[name] for name in node.input]
            if any(a.size != 1 for a in scalars):
                continue
            start, stop, step = (a.item() for a in scalars)
            if (
                not all(math.isfinite(v) for v in (start, stop, step))
                or step == 0
                or max(0, math.ceil((stop - start) / step)) > MAX_CONSTANT_ELEMENTS
            ):
                continue
        try:
            output = ReferenceEvaluator(node, opsets=opsets).run(
                None, {name: values[name] for name in node.input if name}
            )
        except (
            ValueError,
            TypeError,
            RuntimeError,
            NotImplementedError,
            KeyError,
            IndexError,
        ):
            continue
        for name, array in zip(node.output, output, strict=True):
            if (
                name
                and isinstance(array, np.ndarray)
                and array.size <= MAX_CONSTANT_ELEMENTS
            ):
                values[name] = array
    return values


def _fresh(model, stem):
    used = (
        {i.name for i in model.graph.input}
        | {i.name for i in model.graph.initializer}
        | {i.name for i in model.graph.value_info}
        | {name for n in model.graph.node for name in n.output}
    )
    index = 0
    while f"{stem}_{index}" in used:
        index += 1
    return f"{stem}_{index}"


def gather_indices_int32(model, *, allow_dynamic=False):
    """Redirect Gather only; do not change constants used by other consumers.

    Dynamic INT64 casts require explicit opt-in and a caller-established INT32
    range: conversion of out-of-range dynamic inputs can otherwise wrap.
    """
    result = topological_sort(model)
    metadata = _metadata(result)
    values = _constant_values(result)
    initializers = {i.name: i for i in result.graph.initializer}
    inputs = {i.name for i in result.graph.input}
    constant_outputs = {
        name
        for n in result.graph.node
        if not n.domain and n.op_type == "Constant"
        for name in n.output
    }
    replacements, casts = {}, []
    for node in result.graph.node:
        if node.domain or node.op_type != "Gather":
            continue
        name = node.input[1]
        value = values.get(name)
        if value is not None:
            dtype = helper.np_dtype_to_tensor_dtype(value.dtype)
        elif name in initializers:
            dtype = initializers[name].data_type
        elif name in metadata:
            dtype = metadata[name].type.tensor_type.elem_type
        else:
            raise ValueError(f"Unresolved Gather index type: {name}")
        if dtype == TensorProto.INT32:
            continue
        if dtype != TensorProto.INT64:
            raise ValueError(f"Expected integer Gather index: {name}")
        if name not in replacements:
            converted = _fresh(result, "__paraformer_gather_i32")
            if value is not None:
                if np.any(value < np.iinfo(np.int32).min) or np.any(
                    value > np.iinfo(np.int32).max
                ):
                    raise ValueError(f"Gather index outside INT32 range: {name}")
                result.graph.initializer.append(
                    nh.from_array(value.astype(np.int32), converted)
                )
            else:
                if (
                    name in initializers and name not in inputs
                ) or name in constant_outputs:
                    raise ValueError(
                        f"Constant Gather index exceeds supported evaluation budget: {name}"
                    )
                if not allow_dynamic:
                    raise ValueError(
                        f"Dynamic Gather index needs an explicit range contract: {name}"
                    )
                casts.append(
                    helper.make_node(
                        "Cast",
                        [name],
                        [converted],
                        to=TensorProto.INT32,
                        name=converted,
                    )
                )
                # Reserve names before another dynamic replacement is allocated.
                result.graph.value_info.append(
                    helper.make_tensor_value_info(converted, TensorProto.INT32, None)
                )
            replacements[name] = converted
        node.input[1] = replacements[name]
    result.graph.node.extend(casts)
    result = topological_sort(result)
    onnx.checker.check_model(result)
    return result


def fold_constant_ranges(model):
    """Fold statically provable Range values only; no dummy feed or RNG."""
    result = topological_sort(model)
    values = _constant_values(result)
    for node in result.graph.node:
        if node.domain or node.op_type != "Range":
            continue
        if node.output[0] not in values:
            raise ValueError(
                f"Range is data-dependent, unsupported or too large: {node.output[0]}"
            )
        replacement = helper.make_node(
            "Constant",
            [],
            list(node.output),
            name=node.name,
            value=nh.from_array(values[node.output[0]].copy()),
        )
        node.CopyFrom(replacement)
    onnx.checker.check_model(result)
    return result


def normalize_axes(model):
    """Normalize input-rank axes; clone shared axis values per consumer."""
    result = topological_sort(model)
    metadata, values = _metadata(result), _constant_values(result)
    operations = {
        "Split",
        "Softmax",
        "LogSoftmax",
        "Concat",
        "Gather",
        "ReduceMean",
        "ReduceSum",
        "ReduceMax",
        "ReduceMin",
        "Squeeze",
        "Flatten",
    }
    axis_inputs = {"Squeeze", "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin"}
    for node in result.graph.node:
        if node.domain or node.op_type not in operations:
            continue
        shape = _shape(metadata.get(node.input[0]))
        rank = len(shape) if shape is not None else None

        def normalized(axis):
            if rank is None:
                raise ValueError(f"Unknown input rank for {node.name or node.op_type}")
            upper = rank if node.op_type == "Flatten" else rank - 1
            if not -rank <= axis <= upper:
                raise ValueError("Axis outside tensor rank")
            return axis + rank if axis < 0 else axis

        for attr in node.attribute:
            if attr.name == "axis" and attr.i < 0:
                attr.i = normalized(attr.i)
            elif attr.name == "axes" and any(i < 0 for i in attr.ints):
                axes = [normalized(i) for i in attr.ints]
                del attr.ints[:]
                attr.ints.extend(axes)
        if (
            node.op_type in axis_inputs
            and len(node.input) > 1
            and node.input[1] in values
        ):
            axes = values[node.input[1]]
            if np.any(axes < 0):
                if axes.dtype != np.int64 or axes.ndim != 1:
                    raise ValueError("Expected INT64 vector of axes")
                fixed = np.asarray([normalized(int(i)) for i in axes], dtype=np.int64)
                name = _fresh(result, "__paraformer_axes")
                result.graph.initializer.append(nh.from_array(fixed, name))
                node.input[1] = name
    onnx.checker.check_model(result)
    return result
