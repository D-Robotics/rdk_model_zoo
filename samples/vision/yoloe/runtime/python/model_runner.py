# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Reuse the Ultralytics raw runner, with exact YOLOE assets and native output contracts."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from utils.py_utils.assets import sha256_file, verify_asset_file
from utils.py_utils.quantization import dequantize_tensor, validate_scale_quantization
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    BindingError,
    ModelBinding,
    RuntimeMetadata,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import ModelRunner
from samples.vision.ultralytics_yolo.runtime.python.tensor_io import (
    OutputBinding,
    RawOutputs,
    TensorContractError,
    bind_nv12_inputs,
)
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import resolve_platform
from samples.vision.yoloe.runtime.python.model_binding import runtime_selection


def build_runner(selection, *, runtime_loader=None):
    """Load the selected artifact and bind its native input/output descriptors.

    Args:
        selection: Exact published or explicitly identified local float model.
        runtime_loader: Optional SDK-module factory for host injection.

    Returns:
        YOLOERunner: Loaded runner with validated NV12 and output contracts.

    Raises:
        ValueError: If artifact identity, tensor shape/dtype or SCALE metadata
            does not match the selected model.
    """
    selected = runtime_selection(selection)
    if runtime_loader is None:
        from utils.py_utils.platforms import require_execution_target

        require_execution_target(selection.target)
        if selection.local_float:
            if sha256_file(selection.model_path) != selection.local_float_sha256:
                raise ValueError("Local float SHA-256 mismatch.")
        else:
            verify_asset_file(selection.asset, selection.model_path)
    if runtime_loader is None:
        from utils.py_utils.runtime import RuntimeSession
        session = RuntimeSession(str(selection.model_path), target=selection.target)
        session.load()
        model = session.runtime
    else:
        model = runtime_loader().HB_HBMRuntime(str(selection.model_path))
    if len(getattr(model, "model_names", ())) != 1:
        raise BindingError("YOLOE requires exactly one model.")
    metadata = RuntimeMetadata.from_runtime(model)
    binding = bind_yoloe_outputs(selected, metadata, allow_integer=not (
        selection.published_float or selection.local_float))
    return YOLOERunner(model, binding, metadata)


def bind_yoloe_outputs(selection, metadata, *, allow_integer):
    """Bind ten unique NHWC heads by shape and validate their native quantization.

    Integer S heads retain their SCALE metadata until postprocess. Float heads
    remain raw logits; quantization never changes the decoder's sigmoid policy.

    Args:
        selection: Runtime selection carrying the PF11 or PF26 contract.
        metadata: Native SDK names, shapes, dtypes and quantization descriptors.
        allow_integer: Whether this is a published S native-output route.

    Returns:
        ModelBinding: Physical inputs and ten semantic output roles.

    Raises:
        ValueError: On an invalid shape, dtype, NV12 or SCALE descriptor.
    """
    adapter = bind_nv12_inputs(
        resolve_platform(selection.target), metadata.model_name,
        metadata.input_names, metadata.input_shapes, metadata.input_dtypes,
        input_shape_override=(640, 640),
        allow_packed_nhwc=getattr(selection.contract, "allow_packed_nhwc", False),
    )
    box_channels = getattr(selection.contract, "box_channels", 64)
    kinds = (("cls", 4585), ("box", box_channels), ("mces", 32))
    expected = {
        f"{kind}_{stride}": (1, 640 // stride, 640 // stride, channels)
        for stride in (8, 16, 32)
        for kind, channels in kinds
    }
    expected["protos"] = (1, 160, 160, 32)
    if len(metadata.output_names) != 10 or len(set(metadata.output_names)) != 10:
        raise BindingError("YOLOE requires exactly ten unique NHWC outputs.")
    roles = {}
    for role, shape in expected.items():
        names = [n for n in metadata.output_names if metadata.output_shapes.get(n) == shape]
        if len(names) != 1:
            raise BindingError(f"YOLOE role {role} requires one NHWC tensor {shape}.")
        name = roles[role] = names[0]
        dtype = metadata.output_dtypes.get(name)
        if dtype == np.dtype("float32"):
            continue
        expected_dtype = np.dtype("int32")
        if role == "protos":
            expected_dtype = np.dtype("int8" if box_channels == 4 else "int16")
        if not allow_integer or dtype != expected_dtype:
            raise BindingError(f"Unsupported YOLOE dtype for {name}: {dtype}.")
        validate_scale_quantization(metadata.output_quantization.get(name), shape)
    output = YOLOEOutputBinding(
        model_name=metadata.model_name, role_to_name=roles,
        shapes=metadata.output_shapes, dtypes=metadata.output_dtypes,
        expected_shapes=expected, channels={r: sh[-1] for r, sh in expected.items()},
        layouts={r: "NHWC" for r in roles}, runtime_order=metadata.output_names,
        quants=metadata.output_quantization,
    )
    return ModelBinding(selection, selection.contract, metadata, adapter, output)


@dataclass(frozen=True)
class YOLOEOutputBinding(OutputBinding):
    """Validate native SDK arrays, then dequantize at the decode boundary.

    Attributes:
        quants: Physical-output SCALE descriptors for integer heads; float32
            heads retain their native values regardless of vestigial metadata.
    """

    quants: Mapping = field(default_factory=dict)

    def read_raw(self, outputs):
        """Return borrowed role-keyed native arrays without transforming values.

        Args:
            outputs: SDK model/output mapping or this binding's RawOutputs.

        Returns:
            RawOutputs: Arrays retaining their physical NHWC shape and dtype.

        Raises:
            TensorContractError: On foreign bindings, names, shape/dtype or
                nonfinite values.
        """
        if isinstance(outputs, RawOutputs):
            if outputs.binding is not self:
                raise TensorContractError("Raw outputs belong to a different model binding.")
            values = {self.role_to_name[r]: v for r, v in outputs.items()}
        else:
            values = self._unwrap(outputs)
        if not isinstance(values, Mapping) or set(values) != set(self.names):
            raise TensorContractError("YOLOE runtime output names differ from its binding.")
        result = {}
        for role, name in self.role_to_name.items():
            value = np.asarray(values[name])
            if (
                value.shape != self.expected_shapes[role]
                or value.dtype != self.dtypes[name]
                or not np.isfinite(value).all()
            ):
                raise TensorContractError(f"YOLOE output {name} differs from bound shape/dtype or is nonfinite.")
            result[role] = value
        return RawOutputs(result, self)

    def read(self, outputs):
        """Convert integer heads using validated SCALE descriptors.

        Args:
            outputs: SDK outputs or native arrays from this binding's runner.

        Returns:
            dict: Ten semantic NHWC float32 tensors, with logits unchanged in
                meaning and integer heads dequantized by scale/zero-point.

        Raises:
            ValueError: On an invalid native tensor or quantization descriptor.
        """
        raw = self.read_raw(outputs)
        result = {}
        for role, value in raw.items():
            if value.dtype == np.float32:
                result[role] = value
            else:
                info = self.quants[self.role_to_name[role]]
                validate_scale_quantization(info, value.shape)
                result[role] = np.asarray(dequantize_tensor(value, info), dtype=np.float32)
        return result


class YOLOERunner(ModelRunner):
    """Validate physical NV12 input before reusing the common raw SDK call."""

    def __call__(self, tensors):
        if not isinstance(tensors, Mapping) or set(tensors) != {self.model_name}:
            raise ValueError("Expected exactly the bound model input mapping.")
        inputs = tensors[self.model_name]
        if not isinstance(inputs, Mapping) or set(inputs) != set(self.input_names):
            raise ValueError("NV12 input names differ from the binding.")
        for name in self.input_names:
            array = np.asarray(inputs[name])
            shape = (614400,) if self.input_adapter.packed else self.input_shapes[name]
            if (
                array.shape != shape
                or array.dtype != np.uint8
                or not array.flags.c_contiguous
            ):
                raise ValueError(f"{name} requires contiguous uint8 NV12 {shape}.")
        return super().__call__(tensors)
