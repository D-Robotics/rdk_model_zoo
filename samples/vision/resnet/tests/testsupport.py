"""Host-only runtime metadata fixtures for the ResNet pilot tests."""

from __future__ import annotations

from samples.vision.resnet.runtime.python.model_binding import RuntimeMetadata


def runtime_metadata(protocol: str) -> RuntimeMetadata:
    """Return the observed ResNet18 metadata shape for one source family."""

    if protocol == "x5":
        return RuntimeMetadata.from_mapping(
            {
                "model_name": "resnet18_224x224_nv12",
                "input_names": ["data"],
                "input_shapes": {"data": (1, 3, 224, 224)},
                "input_dtypes": {"data": "U8"},
                "output_names": ["prob"],
                "output_shapes": {"prob": (1, 1000, 1, 1)},
                "output_dtypes": {"prob": "F32"},
            }
        )
    return RuntimeMetadata.from_mapping(
        {
            "model_name": "resnet18_224x224_nv12",
            "input_names": ["input_y", "input_uv"],
            "input_shapes": {
                "input_y": (1, 224, 224, 1),
                "input_uv": (1, 112, 112, 2),
            },
            "input_dtypes": {"input_y": "U8", "input_uv": "U8"},
            "output_names": ["output"],
            "output_shapes": {"output": (1, 1000)},
            "output_dtypes": {"output": "F32"},
        }
    )
