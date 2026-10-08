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
## Artifacts (kept material)

<a id="known-gaps"></a>
## Additional preparation

For the selected variant, prepare the ONNX file named by its YAML (`edgenext_<variant>.onnx`) with the expected RGB/NCHW 224x224 input. Prepare `./calibration_data_rgb_f32` as float32 RGB `.npy` data using the YAML mean `123.675 116.28 103.53` and scale `0.01712475 0.017507 0.01742919`. The output prefix carries the variant name, so the emitted `.bin` uses the published filename. Run the checker and compiler commands above with that variant's YAML.
