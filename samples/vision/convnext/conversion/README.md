# ConvNeXt conversion

This directory provides the conversion assets for rebuilding the RDK X5
deployment models of the ConvNeXt sample: three reference PTQ YAMLs
(`ConvNeXt_atto.yaml`, `ConvNeXt_femto.yaml`, `ConvNeXt_nano.yaml`).
The `atto` YAML maps to the manifest-backed variant; `femto` and `nano` are additional build variants. Before compiling, prepare variant-matched ONNX graphs and calibration data at the YAML paths, then run the OE commands below.

<a id="source-model"></a>
## Source model

ConvNeXt (paper [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545),
reference implementation
[facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)).
The atto/femto/nano sizes come from the official ConvNeXt size ladder. The
YAMLs' `onnx_model` entries do **not** line up with their own variant names
(see gap 1); verify the ONNX provenance of every config before compiling,
with the current file values listed in [Additional preparation](#known-gaps).

<a id="directory"></a>
## Directory structure

```text
conversion/
├── ConvNeXt_atto.yaml  # Configuration
├── ConvNeXt_femto.yaml  # Configuration
├── ConvNeXt_nano.yaml  # Configuration
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

Export the selected ConvNeXt variant to ONNX with the official repository
or a compatible pipeline, then set the selected YAML `onnx_model` field
to that graph before running `hb_mapper`. Note that the three YAMLs point at **different and
cross-swapped** files — the atto config consumes
`./convnext_femto.onnx`, the femto config points at
`../../../01_common/model_zoo/mapper/classification/ConvNeXt/convnext_atto.onnx`
(a path outside this sample), and the nano config points at
`./convnext_pico.onnx` (a size not otherwise present in this directory).

<a id="calibration"></a>
## Calibration

All three YAMLs
expect `./calibration_data_rgb_f32` (float32 RGB `.npy`) and use
`calibration_type: 'default'`. Equivalent data must follow the YAML
numerics (mean `123.675 116.28 103.53`, scale `0.01712475 0.017507
0.01742919`, 224x224); this equivalence is a stated requirement, not a
verified pipeline.

<a id="compile"></a>
## Compile

In the OE environment, with both inputs prepared (atto shown), first
verify the ONNX model, then build the deployment model:

```bash
# cwd: this conversion directory
# input: the YAML's onnx_model target (see gap 1 — as delivered this is
#        ./convnext_femto.onnx for the atto config) + ./calibration_data_rgb_f32
# output: working_dir 'ConvNeXt-deploy_224x224_nv12', emitted
#         ConvNeXt-deploy_224x224_nv12.bin — rename required, see gaps
hb_mapper checker --config ConvNeXt_atto.yaml
hb_mapper makertbin --config ConvNeXt_atto.yaml
```

Repeat the same workflow with `ConvNeXt_femto.yaml` or
`ConvNeXt_nano.yaml` to convert the other variants.

All three YAMLs share the deployment protocol `march: bayes-e`, runtime
input `nv12`, training input `rgb` with `NCHW` layout, normalization
`data_mean_and_scale`, and prefix `ConvNeXt-deploy_224x224_nv12`. They set
`compile_mode: 'latency'` / `optimize_level: 'O3'` and place
depthwise/normalization nodes on the BPU with int16 I/O via `node_info`
(8 placements in atto, 5 in femto, 8 in nano; no Softmax placements —
ConvNeXt has no attention softmax). None carries `debug_mode` or
`set_all_nodes_int16`. Before conversion, confirm `onnx_model`,
`cal_data_dir`, `working_dir`, and `output_model_file_prefix` in the
selected YAML.

<a id="validation"></a>
## Validation

Inspect the compiled model on the x86 host:

```bash
# cwd: this conversion directory; input: the emitted .bin (see gap 3 for
# the rename to the manifest name)
hb_perf model_perf \
    --model ./ConvNeXt-deploy_224x224_nv12.bin \
    --input-shape data 1x3x224x224
```

```bash
# cwd: this conversion directory
hrt_model_exec perf \
    --model_file ./ConvNeXt-deploy_224x224_nv12.bin \
    --thread_num 1
```

The functional check on board is the sample runtime
(`python3 samples/vision/convnext/runtime/python/main.py --target x5
--asset-id x5:convnext:ConvNeXt_atto_224x224_nv12.bin ...`, see
[runtime/python/README.md](../runtime/python/README.md)). The expected
runtime protocol is NV12 input at 224x224 and an F32 `1x1000x1x1` output.

<a id="artifacts"></a>
## Artifacts (kept material)

The three YAML files define the variants, input paths and X5 compiler settings summarized above.

<a id="known-gaps"></a>
## Additional preparation

Before compiling, match each graph to its selected YAML and variant:

- The current `onnx_model` values are atto → `./convnext_femto.onnx`, femto → `../../../01_common/model_zoo/mapper/classification/ConvNeXt/convnext_atto.onnx`, and nano → `./convnext_pico.onnx`. Export or select the corresponding atto/femto/nano graph, then set the selected YAML's `onnx_model` to that file.
- Prepare `./calibration_data_rgb_f32` as float32 RGB `.npy` data using `calibration_type: 'default'`, mean `123.675 116.28 103.53`, scale `0.01712475 0.017507 0.01742919`, and 224x224 inputs.
- Each YAML uses `output_model_file_prefix: 'ConvNeXt-deploy_224x224_nv12'`. The emitted file is `ConvNeXt-deploy_224x224_nv12.bin`; use an isolated output directory for each variant and give the selected artifact its manifest filename before deployment. The manifest lists atto as the downloadable variant.
- Run the OE checker and compiler commands above with the YAML that matches the graph and calibration data.
