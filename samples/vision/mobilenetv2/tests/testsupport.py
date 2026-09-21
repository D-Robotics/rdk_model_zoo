"""Host-only runtime metadata fixtures for the MobileNetV2 sample tests.

The tensor names below are synthetic host fixtures; the binding machinery
matches input roles by declared shape, not by name, and the published
artifacts' real names are read from the board at run time.
"""

from __future__ import annotations

from samples.vision.mobilenetv2.runtime.python.model_binding import RuntimeMetadata


def runtime_metadata(protocol: str, wrong_geometry: bool = False, output_dtype: str = "F32", quant_descriptor: bool = False) -> RuntimeMetadata:
    """Return the observed metadata shape for one source family.

    ``wrong_geometry=True`` offsets the declared height/width by eight pixels
    to prove that bind_model rejects metadata that contradicts the contract.
    ``output_dtype``/``quant_descriptor`` reproduce board-observed artifact
    shapes: real X5 mobilenet artifacts ship F32 outputs that still carry a
    vestigial compiler quant descriptor, so the raw_f32 contract must gate on
    dtype and keep the descriptor visible instead of rejecting it.
    """

    height = width = 224
    if wrong_geometry:
        height += 8
        width += 8
    if protocol == "x5":
        return RuntimeMetadata.from_mapping(
            {
                "model_name": "mobilenetv2_224x224_nv12",
                "input_names": ["data"],
                "input_shapes": {"data": (1, 3, height, width)},
                "input_dtypes": {"data": "U8"},
                "output_names": ["prob"],
                "output_shapes": {"prob": (1, 1000, 1, 1)},
                "output_dtypes": {"prob": output_dtype},
                "output_quants": {"prob": {"scale": 0.0078125, "zero_point": -3}} if quant_descriptor else {},
            }
        )
    return RuntimeMetadata.from_mapping(
        {
            "model_name": "mobilenetv2_224x224_nv12",
            "input_names": ["input_y", "input_uv"],
            "input_shapes": {
                "input_y": (1, height, width, 1),
                "input_uv": (1, height // 2, width // 2, 2),
            },
            "input_dtypes": {"input_y": "U8", "input_uv": "U8"},
            "output_names": ["output"],
            "output_shapes": {"output": (1, 1000)},
            "output_dtypes": {"output": output_dtype},
            "output_quants": {"output": {"scale": 0.0078125, "zero_point": -3}} if quant_descriptor else {},
        }
    )
