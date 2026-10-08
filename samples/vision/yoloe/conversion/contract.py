# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Static ONNX/vocabulary checks before any calibration output or compiler call."""

from pathlib import Path
from utils.py_utils.assets import sha256_file
from samples.vision.yoloe.model.vocabulary import LABELS_SHA256


def inspect_graph(path, names_path, variant):
    """Require one static RGB F32 input and ten uniquely identifiable F32 NHWC heads.

    Graph structure validates the protocol, not checkpoint architecture or accuracy.
    External tensor data is rejected: a single-file digest must cover all weights.
    """
    import onnx

    path = Path(path)
    names_path = Path(names_path)
    if sha256_file(names_path) != LABELS_SHA256:
        raise ValueError(
            "Vocabulary checksum mismatch; preserve the fixed ordered PF labels."
        )
    from google.protobuf.message import DecodeError

    try:
        model = onnx.load(str(path), load_external_data=False)
        if any(
            t.data_location == onnx.TensorProto.EXTERNAL
            for t in _tensors(model, onnx.TensorProto)
        ):
            raise ValueError(
                "External ONNX tensor data is unsupported; export a self-contained graph."
            )
        onnx.checker.check_model(model)
    except (DecodeError, onnx.checker.ValidationError) as error:
        raise ValueError(f"Invalid ONNX graph: {error}") from error
    if len(model.graph.input) != 1:
        raise ValueError("YOLOE ONNX requires exactly one input.")
    image = model.graph.input[0]

    def shape(value):
        dims = value.type.tensor_type.shape.dim
        if any(
            dim.HasField("dim_param")
            or not dim.HasField("dim_value")
            or dim.dim_value <= 0
            for dim in dims
        ):
            raise ValueError(f"Static positive dimensions required: {value.name}")
        return tuple(d.dim_value for d in dims)

    if (
        image.name != "images"
        or shape(image) != (1, 3, 640, 640)
        or image.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
    ):
        raise ValueError("Expected ONNX images input float32 [1,3,640,640].")
    expected = {
        f"{kind}_{stride}": (1, 640 // stride, 640 // stride, c)
        for stride in (8, 16, 32)
        for kind, c in [
            ("cls", 4585),
            ("box", 4 if variant.startswith("26") else 64),
            ("mces", 32),
        ]
    }
    expected["protos"] = (1, 160, 160, 32)
    if len(model.graph.output) != 10 or len({v.name for v in model.graph.output}) != 10:
        raise ValueError("Expected ten unique ONNX outputs.")
    roles = {}
    for value in model.graph.output:
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("Every ONNX output must be float32.")
        observed = shape(value)
        matches = [k for k, s in expected.items() if s == observed]
        if len(matches) != 1 or matches[0] in roles:
            raise ValueError(f"Output protocol mismatch: {value.name} {observed}.")
        roles[matches[0]] = value.name
    return {
        "onnx_sha256": sha256_file(path),
        "names_sha256": sha256_file(names_path),
        "input_name": "images",
        "input_shape": [1, 3, 640, 640],
        "output_roles": roles,
        "output_shapes": {k: list(v) for k, v in expected.items()},
        "output_dtype": "float32",
        "softmax_nodes": [
            node.name for node in model.graph.node if node.op_type == "Softmax"
        ],
    }


def _tensors(message, tensor_type):
    """Walk protobuf message fields, including nested functions/graphs/attributes."""
    if isinstance(message, tensor_type):
        yield message
        return
    for descriptor, value in message.ListFields():
        if descriptor.message_type is not None:
            for child in value if descriptor.is_repeated else (value,):
                yield from _tensors(child, tensor_type)
