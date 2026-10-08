# EfficientFormerV2 conversion

This directory provides conversion assets: the three reference
PTQ YAMLs (`EfficientFormerv2_s0_config.yaml`, `EfficientFormerv2_s1_config.yaml`,
`EfficientFormerv2_s2_config.yaml`). Prepare variant-matched ONNX graphs and calibration data at the paths specified by each YAML, then use the OE compile steps below.

<a id="source-model"></a>
## Source model

EfficientFormerV2-S0/S1/S2 (paper [EfficientFormerV2: Rethinking Vision
Transformers for MobileNet Size and
Speed](https://arxiv.org/abs/2212.08059)). The YAMLs expect `./efficientformerv2_s0.onnx`, `./efficientformerv2_s1.onnx`, and `./efficientformerv2_s2.onnx`; export each matching variant to its YAML path.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── EfficientFormerv2_s0_config.yaml  # Configuration
├── EfficientFormerv2_s1_config.yaml  # Configuration
├── EfficientFormerv2_s2_config.yaml  # Configuration
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Run model conversion on an x86 Linux host inside the RDK X5 OpenExplorer
Docker (march `bayes-e`). Prepare the toolchain with `hb_mapper`, `hb_perf`, and `hrt_model_exec`; offline Docker images are available from the D-Robotics developer forum ([topic 35229](https://forum.d-robotics.cc/t/topic/35229)).

<a id="export"></a>
## Export

Export ONNX models for the selected upstream EfficientFormerV2 variant with `timm`:

1. Install the required Python packages such as `timm`, `onnx`, and
   `onnxsim`.
2. Create the target EfficientFormerV2 model with `timm.models.create_model`,
   such as `efficientformerv2_s0`, `efficientformerv2_s1`, or
   `efficientformerv2_s2`.
3. Export the model with `torch.onnx.export` using a `1x3x224x224` dummy
   input.
4. Simplify the ONNX model with `onnxsim.simplify`.
5. Compile the simplified ONNX model in the OE environment (see Compile).

The resulting files must be named `efficientformerv2_s0.onnx` /
`efficientformerv2_s1.onnx` / `efficientformerv2_s2.onnx` and placed in
this directory (or the YAMLs' `onnx_model` adjusted).

<a id="calibration"></a>
## Calibration

All three YAMLs
expect `./calibration_data_rgb_f32` (float32 RGB `.npy`) and use
`calibration_type: 'max'` with per-variant `max_percentile`: S0 and S1
`0.999`, S2 `0.9995`. Apply the YAML numerics to calibration data (mean
`123.675 116.28 103.53`, scale `0.01712475 0.017507 0.01742919`,
224x224).

<a id="compile"></a>
## Compile

In the OE environment, with both inputs prepared, compile the matching
variant:

```bash
# cwd: this conversion directory
# input: ./efficientformerv2_s0.onnx + ./calibration_data_rgb_f32
# output: working_dir 'EfficientFormerv2_s0_int16_model_output', emitted
#         .bin named by output_model_file_prefix (see below)
hb_mapper checker --config EfficientFormerv2_s0_config.yaml
hb_mapper makertbin --config EfficientFormerv2_s0_config.yaml
```

Unlike the efficientnet/efficientformer X5 deliveries, each YAML here
carries its variant identity: `output_model_file_prefix` is
`EfficientFormerv2_s{0,1,2}_224x224_nv12`, so the emitted `.bin` name
reproduces the manifest basenames
(`EfficientFormerv2_s0_224x224_nv12.bin`,...) with no rename step, and
each variant compiles into its own `working_dir` with no collision. All
three YAMLs set `compile_mode: 'latency'` / `optimize_level: 'O3'` and
carry `node_info` Softmax int16 placements (5 nodes for S0/S1, 10 for S2).
S0 additionally sets `debug_mode: "dump_calibration_data"` and
`optimization: "set_all_nodes_int16"`; S1/S2 omit these settings.

<a id="validation"></a>
## Validation

Use the OE package tools `hb_perf` and `hrt_model_exec` for host-side
model inspection. The functional check on board is
the sample runtime:
`python3 samples/vision/efficientformerv2/runtime/python/main.py --target x5 --asset-id x5:efficientformerv2:EfficientFormerv2_s0_224x224_nv12.bin...`
(see [runtime/python/README.md](../runtime/python/README.md)). The runtime
expects an input tensor of `1x3x224x224` before NV12 packing and returns
ImageNet-1k classification logits.

<a id="artifacts"></a>
## Recipe files

The three reference YAMLs are this directory's conversion assets; their
SHA-256 digests are:

| File | SHA-256 |
| --- | --- |
| `EfficientFormerv2_s0_config.yaml` | `a0415f8a4a3f75976be1c8a5a0bf30aa9874b95ee5f61ec10d6ee7147c5d4351` |
| `EfficientFormerv2_s1_config.yaml` | `530d78e7d2e28eb57832b2f8d48d7d1d8f4de5359b127eac915922be6353fcc9` |
| `EfficientFormerv2_s2_config.yaml` | `b35d73d6059f5e7765f415eac1863a72792d0e090d6c04640e1164b4814ad699` |

<a id="known-gaps"></a>
## Additional preparation

Prepare the matching ONNX graph for each `efficientformerv2_s*.onnx` input and float32 RGB `.npy` data under `./calibration_data_rgb_f32`, following the selected YAML's 224x224 normalization (mean `123.675 116.28 103.53`, scale `0.01712475 0.017507 0.01742919`). S0 sets `debug_mode: "dump_calibration_data"` and `optimization: "set_all_nodes_int16"`; S1/S2 omit these settings. S0 uses `working_dir: 'EfficientFormerv2_s0_int16_model_output'`; S1/S2 use `EfficientFormerv2_s{1,2}_224x224_nv12`. Run each variant in its YAML-selected output directory with the commands above.
