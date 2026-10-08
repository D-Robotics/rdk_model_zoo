# RepVGG conversion

<a id="source-model"></a>
## Source model

Use the official RepVGG flow: create the selected model with `create_RepVGG_B1g2(deploy=False)`, run `repvgg_model_convert`, then export ONNX. Record the source revision, PyTorch version and checkpoint digest.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── RepVGG_A0_config.yaml  # Configuration
├── RepVGG_A1_config.yaml  # Configuration
├── RepVGG_A2_config.yaml  # Configuration
├── RepVGG_B0_config.yaml  # Configuration
├── RepVGG_B1g2_config.yaml  # Configuration
└── RepVGG_B1g4_config.yaml  # Configuration
```

<a id="toolchain-targets"></a>
## Toolchain and targets

6 YAML configurations target X5 march `bayes-e`. Record the OE version used for compilation.

| Config | ONNX input path | Working directory | Compiled basename |
| --- | --- | --- | --- |
| `RepVGG_A0_config.yaml` | `./RepVGG-A0.onnx` | `RepVGG-A0_224x224_nv12` | `RepVGG-A0_224x224_nv12.bin` |
| `RepVGG_A1_config.yaml` | `./RepVGG-A1.onnx` | `RepVGG-A1_224x224_nv12` | `RepVGG-A1_224x224_nv12.bin` |
| `RepVGG_A2_config.yaml` | `./RepVGG-A2.onnx` | `RepVGG-A2_224x224_nv12` | `RepVGG-A2_224x224_nv12.bin` |
| `RepVGG_B0_config.yaml` | `./RepVGG-B0.onnx` | `RepVGG-B0_224x224_nv12` | `RepVGG-B0_224x224_nv12.bin` |
| `RepVGG_B1g2_config.yaml` | `./RepVGG-B1g2.onnx` | `RepVGG-B1g2_224x224_nv12` | `RepVGG-B1g2_224x224_nv12.bin` |
| `RepVGG_B1g4_config.yaml` | `./RepVGG-B1g4.onnx` | `RepVGG-B1g4_224x224_nv12` | `RepVGG-B1g4_224x224_nv12.bin` |


Toolchain resources:

- [OE Docker environment](https://forum.d-robotics.cc/t/topic/35229)

<a id="export"></a>
## ONNX export

Use the official RepVGG flow: create the selected model with `create_RepVGG_B1g2(deploy=False)`, run `repvgg_model_convert`, then export ONNX. Record the source revision, PyTorch version and checkpoint digest.

Export the matching graph to the table path with nominal RGB/NCHW 1×3×224×224 input and ImageNet-1k output. YAML `input_shape` and `input_name` are empty, so inspect the graph for its actual dimensions and names.

### ONNX export example

Run this example in the x86 export environment with PyTorch installed. The simplification step also requires `onnxsim`. Match the exported filename to `onnx_model` in the selected YAML.

Export uses the official RepVGG implementation. Install the example dependencies:

```shell
pip install timm onnx
```

From the Model Zoo repository root, clone the RepVGG source:

```shell
git clone https://github.com/DingXiaoH/RepVGG.git
```

Obtain `RepVGG-B1g2-train.pth` from the pretrained-model links in the [official RepVGG repository](https://github.com/DingXiaoH/RepVGG) and place it in the cloned `RepVGG/` directory, or set `model_path` below to its absolute path. Save this code as `RepVGG/export_onnx.py`.

This example exports RepVGG-B1g2. For another variant, select its matching constructor and checkpoint:

```Python
from repvgg import *
import torch
import onnx
import torch.onnx

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
    model_path = "RepVGG-B1g2-train.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_RepVGG_B1g2(deploy=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = repvgg_model_convert(model)

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "RepVGG-B1g2.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        verbose=True,
        input_names=["data"],  # input name
        output_names=["output"],  # output name
    )

    param_count = count_parameters(onnx_file_path)
    print(f'Total number of parameters: {param_count}')
```

Run from the Model Zoo repository root, then copy the exported ONNX into the conversion directory:

```bash
cd RepVGG
python3 export_onnx.py
cp RepVGG-B1g2.onnx ../samples/vision/repvgg/conversion/
cd ..
```

<a id="calibration"></a>
## Calibration

All YAMLs require `./calibration_data_rgb_f32` (float32, calibration default), RGB/NCHW training input and NV12 runtime input. Mean 123.675/116.28/103.53; scale 0.01712475/0.017507/0.01742919. Dataset selection/count, preparation script and actual calibration files are missing. Check prepared float semantics against the graph and normalization; changing an NV12 directory name does not produce RGB calibration data.

<a id="compile"></a>
## Compile

The A0 commands below require `RepVGG-A0.onnx`. For the B1g2 example above, use `./RepVGG-B1g2.onnx` with checker and `RepVGG_B1g2_config.yaml` with makertbin.

In the OE environment, after the ONNX graph and calibration data
prerequisites are supplied:

```bash
# cwd: repository root, then conversion directory
cd samples/vision/repvgg/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./RepVGG-A0.onnx
hb_mapper makertbin --model-type onnx --config RepVGG_A0_config.yaml
```

Expected output for this config: `RepVGG-A0_224x224_nv12/RepVGG-A0_224x224_nv12.bin`. All configs use latency/O3. Each variant has a distinct directory/prefix, using `RepVGG-A0` etc. Compiled basenames use a hyphen while the published filenames use an underscore. Every YAML also removes Quantize/Dequantize/Transpose/Cast/Reshape nodes and enables dump_calibration_data. Verify graph semantics after these transformations; compare the transformed graph outputs with the floating model.

<a id="validation"></a>
## Post-conversion validation

Before inference, verify the packed NV12 geometry 224×224 and an F32
score output squeezing to (1000,). Example for the first variant:

```bash
# cwd: repository root on X5
python3 samples/vision/repvgg/runtime/python/main.py --target x5 \
  --asset-id x5:repvgg:RepVGG_A0_224x224_nv12.bin \
  --model-path samples/vision/repvgg/conversion/RepVGG-A0_224x224_nv12/RepVGG-A0_224x224_nv12.bin \
  --test-img samples/vision/repvgg/test_data/gooze.JPEG
```

Use the qualified reference that matches the artifact and target. Record the build hash and graph provenance, compare source and unified raw outputs, and evaluate accuracy before delivery.

<a id="artifacts"></a>
## Artifacts

Compiled paths are listed above. Published files and per-variant targets are listed in [model preparation](../model/README.md#artifacts); downloads land in the sample model directory. Store each compiled model with its variant name and compilation configuration.

<a id="known-gaps"></a>
## Additional preparation

Load the matching RepVGG training checkpoint, create the selected model with `deploy=False`, and run `repvgg_model_convert` before ONNX export. Export the graph at the path in the selected YAML with RGB/NCHW 1×3×224×224 input and ImageNet-1k output; inspect input names and shapes before compilation. Prepare `./calibration_data_rgb_f32` as float32 RGB data using mean `123.675/116.28/103.53` and scale `0.01712475/0.017507/0.01742919`. Each YAML writes a variant-specific directory; use the manifest filename when saving the deployment artifact.
