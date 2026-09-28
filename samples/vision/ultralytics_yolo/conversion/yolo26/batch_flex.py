"""Make YOLO26's fixed spatial attention reshapes accept calibration batches.

The exported model keeps a static batch-1 input/output contract. Only named
attention Reshape constants whose leading dimension is the exported batch are
changed, so the compiler can infer that dimension from a calibration batch.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Tuple


TASKS = {"detect", "cls", "seg", "pose", "obb"}


@dataclass(frozen=True)
class BatchFlexReport:
    task: str
    status: str
    patched_nodes: Tuple[str, ...]
    input_shapes: Tuple[Tuple[int, ...], ...]
    output_shapes: Tuple[Tuple[int, ...], ...]


def _value_shape(value: Any) -> Tuple[int, ...]:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        if getattr(dim, "dim_param", ""):
            raise ValueError("dynamic tensor dimensions are not accepted")
        size = int(getattr(dim, "dim_value", 0))
        if size <= 0:
            raise ValueError("tensor dimensions must be statically positive")
        dims.append(size)
    return tuple(dims)


def _is_attention_reshape(node: Any) -> bool:
    name = re.sub(r"[^a-z0-9]+", "/", str(getattr(node, "name", "")).lower())
    return node.op_type == "Reshape" and any(
        part in {"attn", "attention"} for part in name.split("/")
    )


def _read_shape(source: Tuple[str, Any], np_module: Any, numpy_helper: Any):
    kind, value = source
    if kind == "ints":
        return np_module.asarray(list(value.ints), dtype=np_module.int64)
    return numpy_helper.to_array(value)


def _write_shape(source: Tuple[str, Any], shape: Any, numpy_helper: Any) -> None:
    kind, value = source
    if kind == "ints":
        del value.ints[:]
        value.ints.extend(int(item) for item in shape.tolist())
        return
    replacement = numpy_helper.from_array(shape, name=getattr(value, "name", ""))
    value.CopyFrom(replacement)


def adapt_calibration_batch8(
    onnx_path: str | Path,
    task: str,
    *,
    onnx_module: Any = None,
    numpy_module: Any = None,
) -> BatchFlexReport:
    """Patch fixed batch dimensions in an exported YOLO26 ONNX graph in place.

    A missing or unfamiliar target is an error: a no-op must not be reported as
    batch-8 compatible. This structural rewrite does not replace compiler
    evidence that calibration actually ran with batch 8.
    """

    if task not in TASKS:
        raise ValueError(f"unsupported YOLO26 task {task!r}")
    if onnx_module is None:
        try:
            import onnx as onnx_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError("onnx is required to adapt the YOLO26 export graph") from exc
    if numpy_module is None:
        try:
            import numpy as numpy_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError("numpy is required to adapt the YOLO26 export graph") from exc
    numpy_helper = getattr(onnx_module, "numpy_helper", None)
    if numpy_helper is None:
        try:
            from onnx import numpy_helper  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError("onnx.numpy_helper is required to adapt the YOLO26 export graph") from exc

    path = Path(onnx_path)
    model = onnx_module.load(str(path))
    graph = model.graph
    input_shapes = tuple(_value_shape(value) for value in graph.input)
    output_shapes = tuple(_value_shape(value) for value in graph.output)
    if len(input_shapes) != 1 or len(input_shapes[0]) != 4:
        raise ValueError(f"expected one static NCHW input, found {input_shapes}")
    if input_shapes[0][0] != 1 or input_shapes[0][1] != 3:
        raise ValueError(f"expected a batch-1, 3-channel input, found {input_shapes[0]}")
    if not output_shapes or any(not shape or shape[0] != 1 for shape in output_shapes):
        raise ValueError(f"expected static batch-1 outputs, found {output_shapes}")

    constants = {}
    for initializer in graph.initializer:
        constants[initializer.name] = ("tensor", initializer)
    for node in graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attribute in node.attribute:
            if attribute.name == "value" and getattr(attribute, "t", None) is not None:
                constants[node.output[0]] = ("tensor", attribute.t)
                break
            if attribute.name == "value_ints":
                constants[node.output[0]] = ("ints", attribute)
                break

    candidates = []
    for node in graph.node:
        if not _is_attention_reshape(node) or len(node.input) != 2:
            continue
        source = constants.get(node.input[1])
        if source is None:
            continue
        shape = _read_shape(source, numpy_module, numpy_helper)
        if shape.ndim != 1 or not numpy_module.issubdtype(shape.dtype, numpy_module.integer):
            raise RuntimeError(
                f"unsupported attention Reshape target for {node.name!r}: "
                f"expected an integer vector, found {shape!r}")
        if shape.size and int(shape[0]) == 1:
            if shape.size != 4:
                raise RuntimeError(
                    f"unsupported fixed-batch attention Reshape {node.name!r}: "
                    f"expected a rank-4 target, found {shape.tolist()}")
            candidates.append((node, source, shape))

    if not candidates:
        raise RuntimeError(
            f"YOLO26 {task}: no recognized fixed-batch attention Reshape was found; "
            "batch-8 compatibility is unverified")

    for node, source, _ in candidates:
        tensor_name = node.input[1]
        consumers = [other for other in graph.node if tensor_name in other.input]
        unsafe = [other.name or other.op_type for other in consumers
                  if not _is_attention_reshape(other)]
        if unsafe:
            raise RuntimeError(
                f"attention shape constant {tensor_name!r} is shared with "
                f"non-attention nodes {unsafe}; refusing an unsafe rewrite")

    patched = []
    written = set()
    for node, source, shape in candidates:
        identity = id(source[1])
        if identity not in written:
            updated = numpy_module.array(shape, copy=True)
            updated[0] = -1
            _write_shape(source, updated, numpy_helper)
            written.add(identity)
        patched.append(str(node.name or node.output[0] or node.op_type))

    onnx_module.checker.check_model(model)
    if tuple(_value_shape(value) for value in graph.input) != input_shapes:
        raise RuntimeError("batch adaptation changed the model input contract")
    if tuple(_value_shape(value) for value in graph.output) != output_shapes:
        raise RuntimeError("batch adaptation changed the model output contract")

    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        onnx_module.save(model, temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

    print(f"YOLO26_BATCH_FLEX task={task} status=patched count={len(patched)}")
    for name in patched:
        print(f"YOLO26_BATCH_FLEX node={name}")
    return BatchFlexReport(task, "patched", tuple(patched), input_shapes, output_shapes)
