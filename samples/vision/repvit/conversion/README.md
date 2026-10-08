# RepViT conversion

<a id="source-model"></a>
## Source model

Use `timm.models.create_model` for repvit_m0_9/m1_0/m1_1, export with PyTorch and simplify with onnxsim. Record the timm/PyTorch versions, source revision and checkpoint digest.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── RepViT_m0_9_config.yaml  # Configuration
├── RepViT_m1_0_config.yaml  # Configuration
└── RepViT_m1_1_config.yaml  # Configuration
```

<a id="toolchain-targets"></a>
## Toolchain and targets

3 YAML configurations target X5 march `bayes-e`. Record the OE version used for compilation.

| Config | ONNX input path | Working directory | Compiled basename |
| --- | --- | --- | --- |
| `RepViT_m0_9_config.yaml` | `./repvit_m0_9.onnx` | `RepViT_224x224_nv12` | `RepViT_224x224_nv12.bin` |
| `RepViT_m1_0_config.yaml` | `./repvit_m1_0.onnx` | `RepViT_224x224_nv12` | `RepViT_224x224_nv12.bin` |
| `RepViT_m1_1_config.yaml` | `./repvit_m1_1.onnx` | `RepViT_224x224_nv12` | `RepViT_224x224_nv12.bin` |


Toolchain resources:

- [OE Docker environment](https://forum.d-robotics.cc/t/topic/35229)

<a id="export"></a>
## ONNX export

Use `timm.models.create_model` for repvit_m0_9/m1_0/m1_1, export with PyTorch and simplify with onnxsim. Record the timm/PyTorch versions, source revision and checkpoint digest.

Export the matching graph to the table path with nominal RGB/NCHW 1×3×224×224 input and ImageNet-1k output. YAML `input_shape` and `input_name` are empty, so inspect the graph for its actual dimensions and names.

### ONNX export example

Run this example in the x86 export environment with PyTorch installed. The simplification step also requires `onnxsim`. Match the exported filename to `onnx_model` in the selected YAML.

The onnx model is transformed using models from the timm library (PyTorch Image Models). Install the required packages using the following command:

```shell
pip install timm onnx
```

Model transformation takes repvit_m0_9 as an example, and the other three models are the same:

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.repvit import repvit_m0_9, repvit_m1_0, repvit_m1_1

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
    model = create_model('repvit_m0_9', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "repvit_m0_9.onnx"

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
        simplified_onnx_file_path = "repvit_m0_9.onnx"
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

All YAMLs require `./calibration_data_rgb_f32` (float32, calibration default), RGB/NCHW training input and NV12 runtime input. Mean 123.675/116.28/103.53; scale 0.01712475/0.017507/0.01742919. Dataset selection/count, preparation script and actual calibration files are missing. Check prepared float semantics against the graph and normalization; changing an NV12 directory name does not produce RGB calibration data.

<a id="compile"></a>
## Compile

In the OE environment, after the ONNX graph and calibration data
prerequisites are supplied:

```bash
# cwd: repository root, then conversion directory
cd samples/vision/repvit/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./repvit_m0_9.onnx
hb_mapper makertbin --model-type onnx --config RepViT_m0_9_config.yaml
```

Expected output for this config: `RepViT_224x224_nv12/RepViT_224x224_nv12.bin`. All configs use latency/O3. All variants share the same working directory and output basename, without a variant suffix; isolate builds to avoid overwriting. YAML removes Quantize/Dequantize/Transpose/Cast/Reshape nodes. Validate the transformed graph rather than assuming these removals preserve semantics for an arbitrary new export.

<a id="validation"></a>
## Post-conversion validation

Before inference, verify the packed NV12 geometry 224×224 and an F32
score output squeezing to (1000,). Example for the first variant:

```bash
# cwd: repository root on X5
python3 samples/vision/repvit/runtime/python/main.py --target x5 \
  --asset-id x5:repvit:RepViT_m0_9_224x224_nv12.bin \
  --model-path samples/vision/repvit/conversion/RepViT_224x224_nv12/RepViT_224x224_nv12.bin \
  --test-img samples/vision/repvit/test_data/yurt.JPEG
```

Use the qualified reference that matches the artifact and target. Record the build hash and graph provenance, compare source and unified raw outputs, and evaluate accuracy before delivery.

<a id="artifacts"></a>
## Artifacts

Compiled paths are listed above. Published files and per-variant targets are listed in [model preparation](../model/README.md#artifacts); downloads land in the sample model directory. Store each compiled model with its variant name and compilation configuration.

<a id="known-gaps"></a>
## Additional preparation

Use `timm.models.create_model` for the selected `repvit_m0_9`, `repvit_m1_0`, or `repvit_m1_1` checkpoint, export with PyTorch, and simplify with `onnxsim`. Place the matching RGB/NCHW 1×3×224×224 graph at the YAML path. Prepare `./calibration_data_rgb_f32` as float32 RGB data with `calibration: default`, mean `123.675/116.28/103.53`, and scale `0.01712475/0.017507/0.01742919`. The YAMLs share one working directory and output prefix; build variants separately and retain the selected identity.
