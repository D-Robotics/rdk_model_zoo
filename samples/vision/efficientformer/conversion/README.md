# EfficientFormer conversion

This directory is a verbatim rdk_x5 @ac11571 delivery: the two reference
PTQ YAMLs (`EfficientFormer_l1_config.yaml`, `EfficientFormer_l3_config.yaml`).
The X5 source ships **no exporter script and no calibration-data producer**,
so this is a reference configuration, not a reproducible flow; the gaps are
listed under [Known gaps](#known-gaps). No conversion was executed during
this migration (the OpenExplorer environment was not run).

<a id="source-model"></a>
## Source model

EfficientFormer-L1 and EfficientFormer-L3 (paper [EfficientFormer:
ImageNet Transformers at MobileNet
Speed](https://arxiv.org/abs/2206.00171)). The YAMLs expect
`./efficientformer_l1.onnx` / `./efficientformer_l3.onnx`, but the source
delivery documents no export recipe and pins no weights — the ONNX
provenance is unverified.

<a id="toolchain-targets"></a>
## Toolchain and targets

Model conversion runs on an x86 Linux host inside the RDK X5 OpenExplorer
Docker (march `bayes-e`), never on the board. The source README points at
the generic OE flow (`hb_mapper checker` / `hb_mapper makertbin` /
`hb_compile`); offline Docker images are available from the D-Robotics
developer forum.

<a id="export"></a>
## Export

No exporter script ships with this delivery. Regenerating the ONNX inputs
requires reproducing the upstream EfficientFormer export yourself; the
resulting files must be named `efficientformer_l1.onnx` /
`efficientformer_l3.onnx` and placed in this directory (or the YAMLs'
`onnx_model` adjusted). This step is unverified here.

<a id="calibration"></a>
## Calibration

No calibration-data producer ships with this delivery. The YAMLs expect
`./calibration_data_rgb_f32` (float32 RGB `.npy`). Equivalent data must
follow the YAML numerics (mean `123.675 116.28 103.53`, scale
`0.01712475 0.017507 0.01742919`, 224x224); this equivalence is a stated
requirement, not a verified pipeline.

<a id="compile"></a>
## Compile

In the OE environment, with both inputs prepared, compile the matching
variant:

```bash
# input: ./efficientformer_l1.onnx + ./calibration_data_rgb_f32
# output prefix: EfficientFormer_224x224_nv12 (variant-less — see gaps)
hb_mapper makertbin --config EfficientFormer_l1_config.yaml
```

Both YAMLs use `calibration_type: 'default'` with
`optimization: "set_all_nodes_int16"` and `compile_mode: 'latency'` /
`optimize_level: 'O3'`; L3 additionally sets `jobs: 64` and carries more
`node_info` Softmax int16 placements than L1.

<a id="validation"></a>
## Validation

No x86 reference script ships with this delivery. The functional check is
the unified runtime on board:
`python3 samples/vision/efficientformer/runtime/python/main.py --target x5 --asset-id x5:efficientformer:EfficientFormer_l1_224x224_nv12.bin ...`
(see [runtime/python/README.md](../runtime/python/README.md)).
**Not run in this migration:** no export, calibration, or compile was
executed; the consistency claims here are static cross-checks of the YAML
contents and filename/prefix agreement with the manifest.

<a id="artifacts"></a>
## Artifacts (kept material)

Both YAML files are kept byte-verbatim from rdk_x5 @ac11571; their
SHA-256 values are pinned by `tests/test_conversion_layout.py`, so any
future edit is caught by the host suite.

<a id="known-gaps"></a>
## Known gaps

As shipped by the source and preserved here:

1. **No ONNX exporter.** Neither `efficientformer_l1.onnx` nor
   `efficientformer_l3.onnx` has a producing script, pinned weights, or a
   documented export recipe.
2. **No calibration-data producer.** `./calibration_data_rgb_f32` has no
   generating script in the source tree.
3. **Variant-less output prefix.** Both YAMLs emit
   `output_model_file_prefix: 'EfficientFormer_224x224_nv12'` (and share
   `working_dir: 'EfficientFormer_224x224_nv12_int16'`), so the compiled
   file does not carry the L1/L3 identity and building the second variant
   into the same working directory collides. To reproduce the manifest
   artifacts, rename the emitted `.bin` to the manifest name
   (`EfficientFormer_l1_224x224_nv12.bin`, ...) or edit the prefix per
   variant first.
4. **No pinned compile command.** The source README points at the generic
   OE flow; the exact command that produced the published `.bin` files is
   not recorded, so reproduction is unverified.
5. **No conversion executed in this migration.**
