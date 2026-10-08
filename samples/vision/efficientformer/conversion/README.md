# EfficientFormer conversion

This directory provides conversion assets: the two reference
PTQ YAMLs (`EfficientFormer_l1_config.yaml`, `EfficientFormer_l3_config.yaml`).
Prepare variant-matched ONNX graphs and calibration data at the paths specified by each YAML, then use the OE compile steps below.

<a id="source-model"></a>
## Source model

EfficientFormer-L1 and EfficientFormer-L3 (paper [EfficientFormer: Vision
Transformers at MobileNet
Speed](https://arxiv.org/abs/2206.01191)). The YAMLs expect `./efficientformer_l1.onnx` and `./efficientformer_l3.onnx`; export the matching model variant to its YAML path.

Additional reference: [arXiv:2206.00171](https://arxiv.org/abs/2206.00171).

<a id="directory"></a>
## Directory structure

```text
conversion/
├── EfficientFormer_l1_config.yaml  # Configuration
├── EfficientFormer_l3_config.yaml  # Configuration
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Run model conversion on an x86 Linux host inside the RDK X5 OpenExplorer
Docker (march `bayes-e`). Prepare the toolchain with
`hb_mapper`, `hb_perf`, and `hrt_model_exec`; offline Docker images are
available from the D-Robotics developer forum
([topic 35229](https://forum.d-robotics.cc/t/topic/35229)).

<a id="export"></a>
## Export

Export ONNX models for the selected upstream EfficientFormer variant with `timm`:

1. Create the target EfficientFormer model with `timm.models.create_model`,
   such as `efficientformer_l1` or `efficientformer_l3`.
2. Export the model with `torch.onnx.export`.
3. Simplify the ONNX model with `onnxsim.simplify`.
4. Compile the simplified ONNX model in the OE environment (see Compile).

The resulting files must be named `efficientformer_l1.onnx` /
`efficientformer_l3.onnx` and placed in this directory (or the YAMLs'
`onnx_model` adjusted).

### ONNX export example

Run this example in the x86 export environment with PyTorch installed. The simplification step also requires `onnxsim`. Match the exported filename to `onnx_model` in the selected YAML.

The onnx model is transformed using models from the timm library (PyTorch Image Models). Install the required packages using the following command:

```shell
pip install timm onnx
```

Model transformation takes efficientformer_l1 as an example, and the other three models are the same:

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.efficientformer import efficientformer_l1, efficientformer_l3

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
    model = create_model('efficientformer_l1', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "efficientformer_l1.onnx"

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
        simplified_onnx_file_path = "efficientformer_l1.onnx"
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

Prepare the calibration directory expected by the YAMLs:
`./calibration_data_rgb_f32` (float32 RGB `.npy`). Prepare those inputs at
224x224 using the YAML values: mean `123.675 116.28 103.53` and scale
`0.01712475 0.017507 0.01742919`.

<a id="compile"></a>
## Compile

In the OE environment, with both inputs prepared, compile the matching
variant:

```bash
# input: ./efficientformer_l1.onnx + ./calibration_data_rgb_f32
# output prefix: EfficientFormer_224x224_nv12 (variant-less — see gaps)
hb_mapper checker --config EfficientFormer_l1_config.yaml
hb_mapper makertbin --config EfficientFormer_l1_config.yaml
```

Both YAMLs use `calibration_type: 'default'` with
`optimization: "set_all_nodes_int16"` and `compile_mode: 'latency'` /
`optimize_level: 'O3'`; L3 additionally sets `jobs: 64` and carries more
`node_info` Softmax int16 placements than L1.

<a id="validation"></a>
## Validation

Use the OE package tools `hb_perf` and `hrt_model_exec` for host-side
model inspection. The functional check on board is
the sample runtime:
`python3 samples/vision/efficientformer/runtime/python/main.py --target x5 --asset-id x5:efficientformer:EfficientFormer_l1_224x224_nv12.bin...`
(see [runtime/python/README.md](../runtime/python/README.md)). The runtime
expects an input tensor of `1x3x224x224` before NV12 packing and returns
ImageNet-1k classification logits.

<a id="artifacts"></a>
## Artifacts

Use `EfficientFormer_l1_config.yaml` or `EfficientFormer_l3_config.yaml`
with its matching ONNX graph and calibration directory.

<a id="known-gaps"></a>
## Additional preparation

Prepare `efficientformer_l1.onnx` or `efficientformer_l3.onnx` from the matching upstream checkpoint and export flow. Use `./calibration_data_rgb_f32` with the RGB/NCHW normalization in the selected YAML (mean `123.675 116.28 103.53`, scale `0.01712475 0.017507 0.01742919`). Both YAMLs share `working_dir: 'EfficientFormer_224x224_nv12_int16'` and `output_model_file_prefix: 'EfficientFormer_224x224_nv12'`; build one variant at a time in an isolated working directory and assign its manifest filename before deployment. Run the OE commands above with the matching YAML.
