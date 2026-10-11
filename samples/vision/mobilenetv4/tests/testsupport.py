"""Host-only runtime metadata fixtures and the published matrix for the MobileNetV4 sample tests.

The tensor names below are synthetic host fixtures; the binding machinery
matches input roles by declared shape, not by name, and the published
artifacts' real names are read from the board at run time.
"""

from __future__ import annotations

from samples.vision.mobilenetv4.runtime.python.cli import RuntimeMetadata

#: variant -> (square input size, shorter-edge resize, OSS directory)
VARIANTS = {
    'small': (224, 256, 'conv-small-224'),
    'medium': (224, 235, 'conv-medium-224'),
    'large': (256, 269, 'conv-large-256'),
}
#: (variant, target) -> manifest filename of every published artifact
PUBLISHED = {
    ('small', 'x5'): 'mobilenetv4_conv_small_bayese_224x224_nv12.bin',
    ('small', 's100'): 's100/mobilenetv4_conv_small_nashe_224x224_nv12.hbm',
    ('small', 's100p'): 's100p/mobilenetv4_conv_small_nashm_224x224_nv12.hbm',
    ('small', 's600'): 's600/mobilenetv4_conv_small_nashp_224x224_nv12.hbm',
    ('medium', 'x5'): 'mobilenetv4_conv_medium_bayese_224x224_nv12.bin',
    ('medium', 's100'): 's100/mobilenetv4_conv_medium_nashe_224x224_nv12.hbm',
    ('medium', 's100p'): 's100p/mobilenetv4_conv_medium_nashm_224x224_nv12.hbm',
    ('medium', 's600'): 's600/mobilenetv4_conv_medium_nashp_224x224_nv12.hbm',
    ('large', 'x5'): 'mobilenetv4_conv_large_bayese_256x256_nv12.bin',
    ('large', 's100'): 's100/mobilenetv4_conv_large_nashe_256x256_nv12.hbm',
    ('large', 's100p'): 's100p/mobilenetv4_conv_large_nashm_256x256_nv12.hbm',
    ('large', 's600'): 's600/mobilenetv4_conv_large_nashp_256x256_nv12.hbm',
}
DEFAULT_VARIANT = 'small'


def runtime_metadata(protocol: str, variant: str = DEFAULT_VARIANT, wrong_geometry: bool = False, output_dtype: str = "F32", quant_descriptor: bool = False) -> RuntimeMetadata:
    """Return the observed metadata shape for one source family and variant.

    ``wrong_geometry=True`` offsets the declared height/width by eight pixels
    to prove that bind_model rejects metadata that contradicts the contract.
    ``output_dtype``/``quant_descriptor`` reproduce board-observed artifact
    shapes: real X5 mobilenet artifacts ship F32 outputs that still carry a
    vestigial compiler quant descriptor, so the raw_f32 contract must gate on
    dtype and keep the descriptor visible instead of rejecting it.
    """

    height = width = VARIANTS[variant][0]
    if wrong_geometry:
        height += 8
        width += 8
    name = f"mobilenetv4_{variant}_{height}x{width}_nv12"
    if protocol == "x5":
        return RuntimeMetadata.from_mapping(
            {
                "model_name": name,
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
            "model_name": name,
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
