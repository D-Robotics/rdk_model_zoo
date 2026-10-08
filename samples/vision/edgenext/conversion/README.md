# EdgeNeXt conversion

This directory provides conversion assets: four reference PTQ
YAMLs (`EdgeNeXt_{base,small,x_small,xx_small}_config.yaml`). Prepare variant-matched ONNX graphs and calibration data at the paths specified by each YAML, then use the OE compile steps below.

For each variant, use the matching YAML, which names its ONNX input and
variant-specific output prefix.


<a id="source-model"></a>
## Source model

EdgeNeXt base/small/x-small/xx-small (paper [EdgeNeXt: Efficiently
Amalgamated CNN-Transformer Architecture for Mobile Vision
Applications](https://arxiv.org/abs/2206.10589), reference implementation
[mmaaz60/EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt)). The YAMLs expect `./edgenext_{base,small,x_small,xx_small}.onnx`; export the selected model variant to the corresponding path.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── EdgeNeXt_base_config.yaml  # Configuration
├── EdgeNeXt_small_config.yaml  # Configuration
├── EdgeNeXt_x_small_config.yaml  # Configuration
├── EdgeNeXt_xx_small_config.yaml  # Configuration
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Run model conversion on an x86 Linux host inside the RDK X5 OpenExplorer
Docker (march `bayes-e`). Prepare the toolchain with `hb_mapper`, `hb_perf`, and `hrt_model_exec`; offline Docker images are available from the D-Robotics developer forum ([topic 35229](https://forum.d-robotics.cc/t/topic/35229)).

<a id="export"></a>
## Export

Export ONNX models for the selected upstream EdgeNeXt variant with `timm`:

1. Create the target EdgeNeXt model with `timm.models.create_model`, such
   as `edgenext_base`, `edgenext_small`, `edgenext_x_small`, or
   `edgenext_xx_small`.
2. Export the model with `torch.onnx.export`.
3. Simplify the ONNX model with `onnxsim.simplify`.
4. Compile the simplified ONNX model in the OE environment (see Compile).

The resulting files must be named `edgenext_<variant>.onnx` and placed in
this directory (or the YAMLs' `onnx_model` adjusted).

### ONNX export example

Run this example in the x86 export environment with PyTorch installed. The simplification step also requires `onnxsim`. Match the exported filename to `onnx_model` in the selected YAML.

The onnx model is transformed using models from the timm library (PyTorch Image Models). Install the required packages using the following command:

```shell
pip install timm onnx
```

Model transformation takes edgenext_small as an example, and the other three models are the same:

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.edgenext import edgenext_small, edgenext_base, edgenext_x_small, edgenext_xx_small

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
    model = create_model('edgenext_xx_small', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "edgenext_xx_small.onnx"

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
        simplified_onnx_file_path = "edgenext_xx_small.onnx"
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

All four YAMLs use `./calibration_data_rgb_f32` (float32 RGB `.npy`) and use
`calibration_type: 'max'` with `max_percentile: 0.999`. Prepare float32 RGB `.npy` data at 224x224 using the YAML values: mean
`123.675 116.28 103.53` and scale `0.01712475 0.017507 0.01742919`.

<a id="compile"></a>
## Compile

In the OE environment, with both inputs prepared (base shown; the other
variants substitute their own config):

```bash
# cwd: this conversion directory
# input: ./edgenext_base.onnx + ./calibration_data_rgb_f32
# output: working_dir 'EdgeNeXt_base_224x224_nv12', emitted
#         EdgeNeXt_base_224x224_nv12.bin — equals the manifest name, no rename
hb_mapper checker --config EdgeNeXt_base_config.yaml
hb_mapper makertbin --config EdgeNeXt_base_config.yaml
```

All four YAMLs set `compile_mode: 'latency'` / `optimize_level: 'O3'` and
place their cross-covariance-attention (`xca`) Softmax nodes on the BPU
with int16 I/O via `node_info` — 3 placements per config (stages 1/2/3);
the xx-small model adds 13 further placements (16 total). None carries
`debug_mode` or `set_all_nodes_int16`.

<a id="validation"></a>
## Validation

Use the OE package tools `hb_perf` and `hrt_model_exec` for host-side
model inspection. The functional check on board is
the sample runtime:
`python3 samples/vision/edgenext/runtime/python/main.py --target x5 --asset-id x5:edgenext:EdgeNeXt_base_224x224_nv12.bin...`
(see [runtime/python/README.md](../runtime/python/README.md)). The runtime
expects an input tensor of `1x3x224x224` before NV12 packing and returns
ImageNet-1k classification logits.

<a id="artifacts"></a>
## Conversion files

<a id="known-gaps"></a>
## Additional preparation

For the selected variant, prepare the ONNX file named by its YAML (`edgenext_<variant>.onnx`) with the expected RGB/NCHW 224x224 input. Prepare `./calibration_data_rgb_f32` as float32 RGB `.npy` data using the YAML mean `123.675 116.28 103.53` and scale `0.01712475 0.017507 0.01742919`. The output prefix carries the variant name, so the emitted `.bin` uses the published filename. Run the checker and compiler commands above with that variant's YAML.
