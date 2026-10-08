English | [简体中文](README_cn.md)

# FasterNet conversion

This directory provides conversion assets: four reference PTQ
YAMLs (`FasterNet_{S,T0,T1,T2}_config.yaml`). Prepare variant-matched ONNX graphs and calibration data at the paths specified by each YAML, then use the OE compile steps below.

All four configs share the output prefix `FasterNet_224x224_nv12`. The `working_dir` values are S `model_output`, T0 `FasterNet_224x224_nv12_mix`, and T1/T2 `FasterNet_224x224_nv12`; build variants in separate directories.

<a id="source-model"></a>
## Source model

FasterNet S/T0/T1/T2 (paper [Run, Don't Walk: Chasing Higher FLOPS for
Faster Neural Networks](https://arxiv.org/abs/2303.03667)). The YAMLs expect `./fasternet_{s,t0,t1,t2}.onnx`; export the selected variant to the matching path.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── FasterNet_S_config.yaml  # Configuration
├── FasterNet_T0_config.yaml  # Configuration
├── FasterNet_T1_config.yaml  # Configuration
├── FasterNet_T2_config.yaml  # Configuration
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Run model conversion on an x86 Linux host inside the RDK X5 OpenExplorer
Docker (march `bayes-e`). Prepare the toolchain with `hb_mapper`, `hb_perf`, and `hrt_model_exec`; offline Docker images are available from the D-Robotics developer forum ([topic 35229](https://forum.d-robotics.cc/t/topic/35229)).

<a id="export"></a>
## Export

The original FasterNet flow uses the
official FasterNet source code to export ONNX models:

1. Obtain the official FasterNet source code and pretrained weights from
   the reference repository.
2. Create the target FasterNet model, such as `fasternet_t0`,
   `fasternet_t1`, `fasternet_t2`, or `fasternet_s`.
3. Export the model with `torch.onnx.export` using a `1x3x224x224` dummy
   input.
4. Simplify the ONNX model with `onnxsim.simplify`.
5. Compile the simplified ONNX model in the OE environment (see Compile).

The resulting files must be named `fasternet_<variant>.onnx` (lowercase,
as the YAMLs expect) and placed in this directory (or the YAMLs'
`onnx_model` adjusted).

### ONNX export example

Run this example in the x86 export environment with PyTorch installed. The simplification step also requires `onnxsim`. Match the exported filename to `onnx_model` in the selected YAML.

The onnx model is transformed using models from the timm library (PyTorch Image Models). Install the required packages using the following command:

```shell
pip install timm onnx
```

Download the model source code using the following command:

```shell
git clone https://github.com/JierunChen/FasterNet.git
```

Model transformation takes fasternet_t2 as an example, and the other three models are the same:

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify

from models.fasternet import *

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
    model_path = "fasternet_t2-epoch.289-val_acc1.78.8860.pth"
    model = fasternet_t2()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # print(model)

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "fasternet_t2.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        verbose=True,
        input_names=["data"],  # input name
        output_names=["output"],  # output name
    )

    # Simplify the ONNX model
    model_simp, check = simplify(onnx_file_path)

    if check:
        print("Simplified model is valid.")
        simplified_onnx_file_path = "fasternet_t2.onnx"
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
`calibration_type: 'default'`. Equivalent data must follow the YAML
numerics (mean `123.675 116.28 103.53`, scale `0.01712475 0.017507
0.01742919`, 224x224).

<a id="compile"></a>
## Compile

In the OE environment, with both inputs prepared (S shown; the other
variants substitute their own config):

```bash
# cwd: this conversion directory
# input: ./fasternet_s.onnx + ./calibration_data_rgb_f32
# output: working_dir 'model_output' (S) / 'FasterNet_224x224_nv12_mix' (T0)
#         / 'FasterNet_224x224_nv12' (T1/T2), emitted
#         FasterNet_224x224_nv12.bin — rename required, see gaps
hb_mapper checker --config FasterNet_S_config.yaml
hb_mapper makertbin --config FasterNet_S_config.yaml
```

All four YAMLs set `compile_mode: 'latency'` / `optimize_level: 'O3'`.
Only the T0 config places nodes on the BPU with int16 I/O via `node_info`
(2 partial-conv-related placements); S/T1/T2 carry no placements. None
carries `debug_mode` or `set_all_nodes_int16`.

<a id="validation"></a>
## Validation

Use the OE package tools `hb_perf` and `hrt_model_exec` for host-side
model inspection. The functional check on board is
the sample runtime:
`python3 samples/vision/fasternet/runtime/python/main.py --target x5 --asset-id x5:fasternet:FasterNet_S_224x224_nv12.bin...`
(see [runtime/python/README.md](../runtime/python/README.md)). The runtime
expects an input tensor of `1x3x224x224` before NV12 packing and returns
ImageNet-1k classification logits.

<a id="artifacts"></a>
## Recipe files

The four reference YAMLs are this directory's conversion assets; their
SHA-256 digests are:

| File | SHA-256 |
| --- | --- |
| `FasterNet_S_config.yaml` | `f0455d5ec5b1c2b4d63f5c153b14060a1b3c17d9b02f3fbfab848239a040e867` |
| `FasterNet_T0_config.yaml` | `c62dd1daedf245e826dcea215ac7adec4dddc5b654b3092c6c44be67d93b371a` |
| `FasterNet_T1_config.yaml` | `e4a123c23edeb38e6835215ea814a1997082bcc0889ea96dfd93fe8b703a0f7a` |
| `FasterNet_T2_config.yaml` | `ad65f79a6e74191d17da60b727416f9c824e8597cfa4fc4d0112597aae1dd944` |

<a id="known-gaps"></a>
## Additional preparation

Prepare the ONNX graph for the selected `fasternet_<variant>.onnx` and RGB float32 calibration data under `./calibration_data_rgb_f32`, using the YAML's 224x224 input and normalization values. All four YAMLs share `output_model_file_prefix: 'FasterNet_224x224_nv12'`, while their `working_dir` values are `model_output` (S), `FasterNet_224x224_nv12_mix` (T0), and `FasterNet_224x224_nv12` (T1/T2). Build variants in separate directories and save each output with its manifest filename. Run the OE commands above with the matching YAML.
