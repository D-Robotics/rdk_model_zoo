"""Host-only runtime metadata fixtures for the EdgeNeXt sample tests.

The tensor names below are synthetic host fixtures; the binding machinery
matches input roles by declared shape, not by name, and the published
artifacts' real names are read from the board at run time.  Geometry follows
the per-variant contract (L1/L3 are both 224).
"""

from __future__ import annotations

from samples.vision.edgenext.runtime.python.model_binding import RuntimeMetadata


def runtime_metadata(
    protocol: str = "x5",
    size: int = 224,
    wrong_geometry: bool = False,
    output_dtype: str = "F32",
    quant_descriptor: bool = False,
) -> RuntimeMetadata:
    """Return the observed metadata shape for the X5 source family.

    ``wrong_geometry=True`` offsets the declared height/width by eight
    pixels to prove that bind_model rejects metadata that contradicts the
    contract.  ``output_dtype``/``quant_descriptor`` reproduce
    board-observed artifact shapes: real X5 classification artifacts ship
    F32 outputs that still carry a vestigial compiler quant descriptor, so
    the raw_f32 contract must gate on dtype and keep the descriptor visible
    instead of rejecting it.
    """

    assert protocol == "x5", "EdgeNeXt publishes X5 assets only"
    height = width = size
    if wrong_geometry:
        height += 8
        width += 8
    return RuntimeMetadata.from_mapping(
        {
            "model_name": "edgenext",
            "input_names": ["data"],
            "input_shapes": {"data": (1, 3, height, width)},
            "input_dtypes": {"data": "U8"},
            "output_names": ["prob"],
            "output_shapes": {"prob": (1, 1000, 1, 1)},
            "output_dtypes": {"prob": output_dtype},
            "output_quants": {"prob": {"scale": 0.0078125, "zero_point": -3}} if quant_descriptor else {},
        }
    )
