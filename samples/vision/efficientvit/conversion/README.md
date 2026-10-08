English | [简体中文](README_cn.md)

# EfficientViT conversion

This directory provides conversion assets: the single reference
PTQ YAML (`EfficientViT_MSRA_m5_config.yaml`). Prepare variant-matched ONNX graphs and calibration data at the paths specified by each YAML, then use the OE compile steps below.

<a id="source-model"></a>
## Source model

EfficientViT-MSRA m5 (paper [EfficientViT: Memory Efficient Vision
Transformer with Cascaded Group
Attention](https://arxiv.org/abs/2305.07027), reference implementation
[microsoft/Cream/EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT)).
The YAML expects the EfficientViT-MSRA m5 ONNX graph at `./efficientvit_m5.onnx`.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── EfficientViT_MSRA_m5_config.yaml  # Configuration
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Run model conversion on an x86 Linux host inside the RDK X5 OpenExplorer
Docker (march `bayes-e`). Prepare the toolchain with `hb_mapper`, `hb_perf`, and `hrt_model_exec`; offline Docker images are available from the D-Robotics developer forum ([topic 35229](https://forum.d-robotics.cc/t/topic/35229)).

<a id="export"></a>
## Export

Export ONNX from the `timm` implementation for EfficientViT_MSRA:

1. Install the required packages such as `timm`, `onnx`, and `onnxsim`.
2. Create the pretrained `efficientvit_m5` model.
3. Export the model with `torch.onnx.export` using a `1x3x224x224` dummy
   input.
4. Simplify the exported ONNX model with `onnxsim`.
5. Compile the simplified ONNX model in the OE environment (see Compile).

Name the resulting file `efficientvit_m5.onnx` and place it in this directory, or set the YAML `onnx_model` to its actual path.

<a id="calibration"></a>
## Calibration

Prepare `./calibration_data_rgb_f32` as float32 RGB `.npy` data for 224x224
inputs. The YAML uses `calibration_type: 'max'` and
`max_percentile: 0.99999`; use its normalization values: mean
`123.675 116.28 103.53`, scale `0.01712475 0.017507 0.01742919`.

<a id="compile"></a>
## Compile

In the OE environment, with both inputs prepared:

```bash
# cwd: this conversion directory
# input: ./efficientvit_m5.onnx + ./calibration_data_rgb_f32
# output: working_dir 'EfficientViT_msra_224x224_nv12', emitted
#         EfficientViT_msra_224x224_nv12.bin; deploy with the manifest filename
hb_mapper checker --config EfficientViT_MSRA_m5_config.yaml
hb_mapper makertbin --config EfficientViT_MSRA_m5_config.yaml
```

The YAML sets `compile_mode: 'latency'` / `optimize_level: 'O3'` and places
28 attention `Softmax` nodes on the BPU with int16 I/O via `node_info` (the
cascaded-group-attention structure). It carries no `debug_mode` and no
`set_all_nodes_int16` optimization.

<a id="validation"></a>
## Validation

Use the OE package tools `hb_perf` and `hrt_model_exec` for host-side
model inspection. The functional check on board is
the sample runtime:
`python3 samples/vision/efficientvit/runtime/python/main.py --target x5 --asset-id x5:efficientvit:EfficientViT_m5_224x224_nv12.bin...`
(see [runtime/python/README.md](../runtime/python/README.md)). The runtime
expects an input tensor of `1x3x224x224` before NV12 packing and returns
ImageNet-1k classification logits.

<a id="artifacts"></a>
## Recipe files

The reference YAML `EfficientViT_MSRA_m5_config.yaml` is this directory's
conversion asset; its SHA-256 digest is:

| File | SHA-256 |
| --- | --- |
| `EfficientViT_MSRA_m5_config.yaml` | `65915fea82515c66457d1ac48051bd8172f97155e89dc9b9a65499dfcb13312c` |

<a id="known-gaps"></a>
## Additional preparation

Prepare the ONNX graph as `./efficientvit_m5.onnx` and calibration data under `./calibration_data_rgb_f32` as float32 RGB `.npy` for 224x224 input. Use `calibration_type: 'max'`, `max_percentile: 0.99999`, mean `123.675 116.28 103.53`, and scale `0.01712475 0.017507 0.01742919`. The YAML emits `EfficientViT_msra_224x224_nv12.bin`; name the deployed file `EfficientViT_m5_224x224_nv12.bin` to match the manifest. Run the checker and compiler commands above with this YAML.
