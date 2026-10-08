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
