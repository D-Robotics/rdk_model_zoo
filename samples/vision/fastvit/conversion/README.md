English | [简体中文](README_cn.md)

# FastViT conversion

This directory provides conversion assets: four reference PTQ
YAMLs (`FastViT_{S12,SA12,T12,T8}_config.yaml`). Prepare variant-matched ONNX graphs and calibration data at the paths specified by each YAML, then use the OE compile steps below.

The YAML `onnx_model` fields point to the common model-zoo path outside this sample tree, and all four configs share `output_model_file_prefix: 'FastViT_224x224_nv12'`. Prepare each graph at its configured path and name the compiled output for the selected variant.

<a id="source-model"></a>
## Source model

FastViT S12/SA12/T12/T8 (paper [FastViT: A Fast Hybrid Vision Transformer
using Structural
Reparameterization](https://arxiv.org/abs/2303.14189)). The YAMLs consume ONNX graphs from the shared `01_common` model zoo; place the selected FastViT variant at the configured path or update `onnx_model`.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── FastViT_S12_config.yaml  # Configuration
├── FastViT_SA12_config.yaml  # Configuration
├── FastViT_T12_config.yaml  # Configuration
├── FastViT_T8_config.yaml  # Configuration
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Run model conversion on an x86 Linux host inside the RDK X5 OpenExplorer
Docker (march `bayes-e`). Prepare the toolchain with `hb_mapper`, `hb_perf`, and `hrt_model_exec`; offline Docker images are available from the D-Robotics developer forum ([topic 35229](https://forum.d-robotics.cc/t/topic/35229)).

<a id="export"></a>
## Export

Use `timm` to export ONNX for the selected FastViT variant:

1. Create the target FastViT model with `timm.models.create_model`, such
   as `fastvit_t8`, `fastvit_t12`, `fastvit_s12`, or `fastvit_sa12`.
2. Export the model with `torch.onnx.export`.
3. Simplify the ONNX model with `onnxsim.simplify`.
4. Compile the simplified ONNX model in the OE environment (see Compile).

The configs expect their ONNX inputs at
`../../../01_common/model_zoo/mapper/classification/FastViT/fastvit_<variant>.onnx`
— a path outside this sample that this repository does not carry.
Regenerating an input requires either restoring that layout or adjusting
`onnx_model`.

### ONNX export example

Run this example in the x86 export environment with PyTorch installed. The simplification step also requires `onnxsim`. Match the exported filename to `onnx_model` in the selected YAML.

The onnx model is transformed using models from the timm library (PyTorch Image Models). Install the required packages using the following command:

```shell
pip install timm onnx
```

Model transformation takes fastvit_t8 as an example, and the other three models are the same:

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.fastvit import fastvit_s12, fastvit_sa12, fastvit_t8, fastvit_t12

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
    model = create_model('fastvit_t8', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "fastvit_t8.onnx"

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
        simplified_onnx_file_path = "fastvit_t8.onnx"
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

In the OE environment, with both inputs prepared (S12 shown; the other
variants substitute their own config):

```bash
# cwd: this conversion directory
# input: the external 01_common ONNX (see gap 1) + ./calibration_data_rgb_f32
# output: working_dir 'FastViT_224x224_nv12_mix', emitted
#         FastViT_224x224_nv12.bin — rename required, see gaps
hb_mapper checker --config FastViT_S12_config.yaml
hb_mapper makertbin --config FastViT_S12_config.yaml
```

All four YAMLs set `compile_mode: 'latency'` / `optimize_level: 'O3'` and
place reparametrized-attention/MLP nodes on the BPU with int16 I/O via
`node_info` (5/6/4/10 placements for S12/SA12/T12/T8). None carries
`debug_mode` or `set_all_nodes_int16`.

<a id="validation"></a>
## Validation

Use the OE package tools `hb_perf` and `hrt_model_exec` for host-side
model inspection. The functional check on board is
the sample runtime:
`python3 samples/vision/fastvit/runtime/python/main.py --target x5 --asset-id x5:fastvit:FastViT_S12_224x224_nv12.bin...`
(see [runtime/python/README.md](../runtime/python/README.md)). The runtime
expects an input tensor of `1x3x224x224` before NV12 packing and returns
ImageNet-1k classification logits.

<a id="artifacts"></a>
## Recipe files

The four reference YAMLs are this directory's conversion assets; their
SHA-256 digests are:

| File | SHA-256 |
| --- | --- |
| `FastViT_S12_config.yaml` | `50c5b40ab3d801ad72eae45a6927dcce4074cf48af90d62d0b974236d46eb8d2` |
| `FastViT_SA12_config.yaml` | `612f9e668d2a30549c33d72595bc84846d05c2404960b31276f174b8c6ddc8fe` |
| `FastViT_T12_config.yaml` | `17b23a8dc23423e499d68d0f9ec3cacf5b2184148fdaa710f20a125c7509a5e2` |
| `FastViT_T8_config.yaml` | `79ab7b5478b3978af871feb81c90e70b87838fa65d5d922cc8d14becbf7d6a0f` |

<a id="known-gaps"></a>
## Additional preparation

The YAML ONNX inputs reference `../../../01_common/model_zoo/mapper/classification/FastViT/...`. Prepare the corresponding ONNX graph for the selected S12/SA12/T12/T8 variant at that path, or update `onnx_model` to the local graph. Prepare RGB float32 calibration data at `./calibration_data_rgb_f32` using the YAML normalization. All four YAMLs share `output_model_file_prefix: 'FastViT_224x224_nv12'`; build each variant in an isolated working directory and use its manifest filename for deployment. Run the matching checker/compiler commands above.
