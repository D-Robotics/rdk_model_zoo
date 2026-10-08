# MobileOne conversion

<a id="source-model"></a>
## Source model

Use the upstream apple/ml-mobileone flow: load the matching unfused checkpoint (for example `mobileone_s0_unfused.pth.tar`), apply `reparameterize_model(model)`, then export and simplify ONNX. Record the selected upstream revision, package versions and checkpoint digest for each build.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── MobileOne_S0_config.yaml  # Configuration
├── MobileOne_S1_config.yaml  # Configuration
├── MobileOne_S2_config.yaml  # Configuration
├── MobileOne_S3_config.yaml  # Configuration
├── MobileOne_S4_config.yaml  # Configuration
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets

5 YAML configurations target X5 march `bayes-e`. Record the OE version used for compilation.

| Config | ONNX input path | Working directory | Compiled basename |
| --- | --- | --- | --- |
| `MobileOne_S0_config.yaml` | `./mobileone_s0.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S1_config.yaml` | `./mobileone_s1.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S2_config.yaml` | `./mobileone_s2.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S3_config.yaml` | `./mobileone_s3.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S4_config.yaml` | `./mobileone_s4.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |


Toolchain resources:

- [OE Docker environment](https://forum.d-robotics.cc/t/topic/35229)

<a id="export"></a>
## ONNX export

Use the upstream apple/ml-mobileone flow: load the matching unfused checkpoint (for example `mobileone_s0_unfused.pth.tar`), apply `reparameterize_model(model)`, then export and simplify ONNX. Record the selected upstream revision, package versions and checkpoint digest for each build.

Export the matching graph to the table path with nominal RGB/NCHW 1×3×224×224 input and ImageNet-1k output. YAML `input_shape` and `input_name` are empty, so inspect the graph for its actual dimensions and names.

### ONNX export example

Run this example in the x86 export environment with PyTorch installed. The simplification step also requires `onnxsim`. Match the exported filename to `onnx_model` in the selected YAML.

The onnx model is transformed using models from the timm library (PyTorch Image Models). Install the required packages using the following command:

```shell
pip install timm onnx
```

Download the model source code using the following command:

```shell
git clone https://github.com/apple/ml-mobileone.git
```

Model transformation takes mobileone_s0 as an example, and the other four models are the same:

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify

from mobileone import *

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
    model_path = "mobileone_s0_unfused.pth.tar"
    model = mobileone(variant='s0')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = reparameterize_model(model)

    # print(model)

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "mobileone_s0.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        verbose=True,
        input_names=["data"],  # input name
        output_names=["output"],  # output name
        keep_initializers_as_inputs=True
    )

    # Simplify the ONNX model
    model_simp, check = simplify(onnx_file_path)

    if check:
        print("Simplified model is valid.")
        simplified_onnx_file_path = "mobileone_s0.onnx"
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
cd samples/vision/mobileone/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./mobileone_s0.onnx
hb_mapper makertbin --model-type onnx --config MobileOne_S0_config.yaml
```

Expected output for this config: `MobileOne_224x224_nv12_int8/MobileOne_224x224_nv12.bin`. All configs use latency/O3. All variants share `MobileOne_224x224_nv12_int8` and the `MobileOne_224x224_nv12` prefix; isolate builds and preserve variant identity. YAML requests 32 compiler jobs: size the build host accordingly. No node-removal override or debug_mode is set.

<a id="validation"></a>
## Post-conversion validation

Before inference, verify the packed NV12 geometry 224×224 and an F32
score output squeezing to (1000,). Example for the first variant:

```bash
# cwd: repository root on X5
python3 samples/vision/mobileone/runtime/python/main.py --target x5 \
  --asset-id x5:mobileone:MobileOne_S0_224x224_nv12.bin \
  --model-path samples/vision/mobileone/conversion/MobileOne_224x224_nv12_int8/MobileOne_224x224_nv12.bin \
  --test-img samples/vision/mobileone/test_data/tiger_beetle.JPEG
```

Use the qualified reference that matches the artifact and target. Record the build hash and graph provenance, compare source and unified raw outputs, and evaluate accuracy before delivery.

<a id="artifacts"></a>
## Artifacts

Compiled paths are listed above. Published files and per-variant targets are listed in [model preparation](../model/README.md#artifacts); downloads land in the sample model directory. Store each compiled model with its variant name and compilation configuration.

<a id="known-gaps"></a>
## Additional preparation

For a MobileOne rebuild, load the matching unfused checkpoint (for example `mobileone_s0_unfused.pth.tar`), apply `reparameterize_model(model)`, and export/simplify an RGB/NCHW 1×3×224×224 graph at the path in the selected YAML. Prepare float32 RGB calibration data at `./calibration_data_rgb_f32` using the YAML mean `123.675/116.28/103.53`, scale `0.01712475/0.017507/0.01742919`, and `default` calibration. Use the X5 `bayes-e` configurations and record the actual framework and OE versions when building.
