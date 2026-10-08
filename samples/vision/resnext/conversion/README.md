# ResNeXt conversion

<a id="source-model"></a>
## Source model

ResNeXt source references timm resnext50_32x4d and ONNX simplification, with the export example below. Select and record package versions, weight revisions, and checksums.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── ResNeXt50_32x4d_config.yaml  # Configuration
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Use the X5 OpenExplore toolchain with march `bayes-e`.

| YAML | ONNX path (conversion cwd) | Output path |
| --- | --- | --- |
| `ResNeXt50_32x4d_config.yaml` | `./ResNeXt50_32x4d.onnx` | `ResNeXt50_32x4d_224x224_nv12/ResNeXt50_32x4d_224x224_nv12.bin` |


Toolchain resources:

- [OE Docker environment](https://forum.d-robotics.cc/t/topic/35229)

<a id="export"></a>
## ONNX export

Use the ONNX export example below. Prepare `ResNeXt50_32x4d.onnx` next to its YAML, RGB NCHW 1×3×224×224 nominal input; input_shape/name are empty and read from the graph.

### ONNX export example

Run this example in the x86 export environment with PyTorch installed. The simplification step also requires `onnxsim`. Match the exported filename to `onnx_model` in the selected YAML.

The onnx model is transformed using models from the timm library (PyTorch Image Models). Install the required packages using the following command:

```shell
pip install timm onnx
```

Model transformation takes resnext50_32x4d as an example:

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.resnet import resnext50_32x4d

def count_parameters(onnx_model_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    # Get the initializers (weights in the model)
    initializer = model.graph.initializer

    # Calculate the total number of parameters
    total_params = 0
    for tensor in initializer:
        # Get the dimensions of each weight
        dims = tensor.dims
        # Calculate the number of parameters in this weight (product of all dimensions)
        params = 1
        for dim in dims:
            params *= dim
        total_params += params

    return total_params

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model('resnext50_32x4d', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "resnext50_32x4d.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        verbose=True,
        input_names=["data"],  # Input name
        output_names=["output"],  # Output name
    )

    # Simplify the ONNX model
    model_simp, check = simplify(onnx_file_path)

    if check:
        print("Simplified model is valid.")
        simplified_onnx_file_path = "resnext50_32x4d.onnx"
        onnx.save(model_simp, simplified_onnx_file_path)
        print(f"Simplified model saved to {simplified_onnx_file_path}")
    else:
        print("Simplified model is invalid!")

    onnx_model_path = simplified_onnx_file_path  # Replace with your ONNX model path
    total_params = count_parameters(onnx_model_path)
    print(f"Total number of parameters in the model: {total_params}")
```

<a id="calibration"></a>
## Calibration

YAML expects RGB/NCHW training input, NV12 runtime input; mean 123.675/116.28/103.53, scale 0.01712475/0.017507/0.01742919. ResNeXt uses./calibration_data_rgb_f32, float32, calibration default; no preparation/count/data are provided.

<a id="compile"></a>
## Compile

In the OE environment, after the ONNX graph and calibration data are
prepared:

```bash
# cwd: repository root
cd samples/vision/resnext/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./ResNeXt50_32x4d.onnx
hb_mapper makertbin --model-type onnx --config ResNeXt50_32x4d_config.yaml
```

latency/O3 is kept. Output basenames match the published filenames; treat
a rebuilt `.bin` as equivalent only after comparing graph, bytes and
accuracy against the published artifact.

<a id="validation"></a>
## Post-conversion validation

Verify actual metadata, 224×224 packed NV12 and one F32 output squeezing to 1000 scores. A rebuilt model can be selected with the exact contract reference and an external path; keep its hash/provenance separate from the publisher artifact.

```bash
# cwd: repository root on X5; published-artifact smoke
bash samples/vision/resnext/model/download.sh x5 50_32x4d
python3 samples/vision/resnext/runtime/python/main.py --target x5 --variant 50_32x4d
```

<a id="artifacts"></a>
## Artifacts

Published artifacts and landing paths: [model preparation](../model/README.md#artifacts).

<a id="known-gaps"></a>
## Additional preparation

For inference, use the manifest-backed X5 artifact in [model/README.md](../model/README.md). A rebuild uses `ResNeXt50_32x4d.onnx`, RGB/NCHW 1×3×224×224 input, float32 RGB calibration data at `./calibration_data_rgb_f32`, mean `123.675/116.28/103.53`, scale `0.01712475/0.017507/0.01742919`, and the X5 `bayes-e` configuration described above. Select a matching upstream weight and OE toolchain before building.
