# EfficientFormerV2 conversion

This directory is a verbatim rdk_x5 @ac11571 delivery: the three reference
PTQ YAMLs (`EfficientFormerv2_s0_config.yaml`, `EfficientFormerv2_s1_config.yaml`,
`EfficientFormerv2_s2_config.yaml`). The X5 source ships **no exporter
script and no calibration-data producer**, so this is a reference
configuration, not a reproducible flow; the gaps are listed under
[Known gaps](#known-gaps). No conversion was executed during this migration
(the OpenExplorer environment was not run).

<a id="source-model"></a>
## Source model

EfficientFormerV2-S0/S1/S2 (paper [EfficientFormerV2: Rethinking Vision
Transformers for MobileNet Size and
Speed](https://arxiv.org/abs/2212.08059)). The YAMLs expect
`./efficientformerv2_s0.onnx` / `./efficientformerv2_s1.onnx` /
`./efficientformerv2_s2.onnx`, but the source delivery documents no export
recipe and pins no weights — the ONNX provenance is unverified.

<a id="toolchain-targets"></a>
## Toolchain and targets

Model conversion runs on an x86 Linux host inside the RDK X5 OpenExplorer
Docker (march `bayes-e`), never on the board. The source README points at
the generic OE flow (`hb_mapper makertbin`); offline Docker images are
available from the D-Robotics developer forum.

<a id="export"></a>
## Export

No exporter script ships with this delivery. Regenerating the ONNX inputs
requires reproducing the upstream EfficientFormerV2 export yourself; the
resulting files must be named `efficientformerv2_s0.onnx` /
`efficientformerv2_s1.onnx` / `efficientformerv2_s2.onnx` and placed in
this directory (or the YAMLs' `onnx_model` adjusted). This step is
unverified here.

<a id="calibration"></a>
## Calibration

No calibration-data producer ships with this delivery. All three YAMLs
expect `./calibration_data_rgb_f32` (float32 RGB `.npy`) and use
`calibration_type: 'max'` with per-variant `max_percentile`: S0 and S1
`0.999`, S2 `0.9995`. Equivalent data must follow the YAML numerics (mean
`123.675 116.28 103.53`, scale `0.01712475 0.017507 0.01742919`, 224x224);
this equivalence is a stated requirement, not a verified pipeline.

<a id="compile"></a>
## Compile

In the OE environment, with both inputs prepared, compile the matching
variant:

```bash
# cwd: this conversion directory
# input: ./efficientformerv2_s0.onnx + ./calibration_data_rgb_f32
# output: working_dir 'EfficientFormerv2_s0_int16_model_output', emitted
#         .bin named by output_model_file_prefix (see below)
hb_mapper makertbin --config EfficientFormerv2_s0_config.yaml
```

Unlike the efficientnet/efficientformer X5 deliveries, each YAML here
carries its variant identity: `output_model_file_prefix` is
`EfficientFormerv2_s{0,1,2}_224x224_nv12`, so the emitted `.bin` name
reproduces the manifest basenames
(`EfficientFormerv2_s0_224x224_nv12.bin`, ...) with no rename step, and
each variant compiles into its own `working_dir` with no collision. All
three YAMLs set `compile_mode: 'latency'` / `optimize_level: 'O3'` and
carry `node_info` Softmax int16 placements (5 nodes for S0/S1, 10 for S2).
S0 additionally sets `debug_mode: "dump_calibration_data"` and
`optimization: "set_all_nodes_int16"`, which S1/S2 do not — an asymmetry
preserved from the source, recorded as-is.

<a id="validation"></a>
## Validation

No x86 reference script ships with this delivery. The functional check is
the unified runtime on board:
`python3 samples/vision/efficientformerv2/runtime/python/main.py --target x5 --asset-id x5:efficientformerv2:EfficientFormerv2_s0_224x224_nv12.bin ...`
(see [runtime/python/README.md](../runtime/python/README.md)).
**Not run in this migration:** no export, calibration, or compile was
executed; the consistency claims here are static cross-checks of the YAML
contents and prefix/filename agreement with the manifest.

<a id="artifacts"></a>
## Artifacts (kept material)

The three YAML files are kept byte-verbatim from rdk_x5 @ac11571; their
SHA-256 values are pinned by `tests/test_conversion_layout.py`, so any
future edit is caught by the host suite.

<a id="known-gaps"></a>
## Known gaps

As shipped by the source and preserved here:

1. **No ONNX exporter.** None of the three `efficientformerv2_s*.onnx`
   inputs has a producing script, pinned weights, or a documented export
   recipe.
2. **No calibration-data producer.** `./calibration_data_rgb_f32` has no
   generating script in the source tree.
3. **S0-only debug/optimization asymmetry.** Only the S0 YAML sets
   `debug_mode: "dump_calibration_data"` and
   `optimization: "set_all_nodes_int16"`; the reason for the asymmetry is
   not documented in the source. S0's `working_dir` spelling
   (`EfficientFormerv2_s0_int16_model_output`) also diverges from its
   siblings' (`EfficientFormerv2_s{1,2}_224x224_nv12`) — cosmetic here,
   since the output prefixes still carry the variant identity.
4. **No pinned compile command.** The source README points at the generic
   OE flow; the exact command that produced the published `.bin` files is
   not recorded, so reproduction is unverified.
5. **No conversion executed in this migration.**
