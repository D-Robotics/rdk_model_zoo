"""Host-only runtime metadata fixtures and the published matrix for the ResNet sample tests.

The tensor names below are synthetic host fixtures; the binding machinery
matches input roles by declared shape, not by name, and the published
artifacts' real names are read from the board at run time.
"""

from __future__ import annotations

from samples.vision.resnet.runtime.python.cli import RuntimeMetadata

#: variant -> (square input size, shorter-edge resize, OSS directory)
VARIANTS = {
    '18': (224, 235, '18-224'),
    '34': (224, 235, '34-224'),
    '50': (224, 235, '50-224'),
    '101': (224, 235, '101-224'),
    '152': (224, 235, '152-224'),
}
#: (variant, target) -> manifest filename of every published artifact
PUBLISHED = {
    ('18', 'x5'): 'resnet18_bayese_224x224_nv12.bin',
    ('18', 's100'): 's100/resnet18_nashe_224x224_nv12.hbm',
    ('18', 's100p'): 's100p/resnet18_nashm_224x224_nv12.hbm',
    ('18', 's600'): 's600/resnet18_nashp_224x224_nv12.hbm',
    ('34', 'x5'): 'resnet34_bayese_224x224_nv12.bin',
    ('34', 's100'): 's100/resnet34_nashe_224x224_nv12.hbm',
    ('34', 's100p'): 's100p/resnet34_nashm_224x224_nv12.hbm',
    ('34', 's600'): 's600/resnet34_nashp_224x224_nv12.hbm',
    ('50', 'x5'): 'resnet50_bayese_224x224_nv12.bin',
    ('50', 's100'): 's100/resnet50_nashe_224x224_nv12.hbm',
    ('50', 's100p'): 's100p/resnet50_nashm_224x224_nv12.hbm',
    ('50', 's600'): 's600/resnet50_nashp_224x224_nv12.hbm',
    ('101', 'x5'): 'resnet101_bayese_224x224_nv12.bin',
    ('101', 's100'): 's100/resnet101_nashe_224x224_nv12.hbm',
    ('101', 's100p'): 's100p/resnet101_nashm_224x224_nv12.hbm',
    ('101', 's600'): 's600/resnet101_nashp_224x224_nv12.hbm',
    ('152', 'x5'): 'resnet152_bayese_224x224_nv12.bin',
    ('152', 's100'): 's100/resnet152_nashe_224x224_nv12.hbm',
    ('152', 's100p'): 's100p/resnet152_nashm_224x224_nv12.hbm',
    ('152', 's600'): 's600/resnet152_nashp_224x224_nv12.hbm',
}
DEFAULT_VARIANT = '18'


def runtime_metadata(protocol: str, variant: str = DEFAULT_VARIANT, wrong_geometry: bool = False, output_dtype: str = "F32", quant_descriptor: bool = False) -> RuntimeMetadata:
    """Return the observed metadata shape for one source family and variant.

    ``wrong_geometry=True`` offsets the declared height/width by eight pixels
    to prove that bind_model rejects metadata that contradicts the contract.
    ``output_dtype``/``quant_descriptor`` reproduce board-observed artifact
    shapes: real X5 classification artifacts ship F32 outputs that still carry a
    vestigial compiler quant descriptor, so the raw_f32 contract must gate on
    dtype and keep the descriptor visible instead of rejecting it.
    """

    height = width = VARIANTS[variant][0]
    if wrong_geometry:
        height += 8
        width += 8
    name = f"resnet_{variant}_{height}x{width}_nv12"
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
