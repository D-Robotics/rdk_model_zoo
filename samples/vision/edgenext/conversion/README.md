# EdgeNeXt conversion

This directory is a verbatim rdk_x5 @ac11571 delivery: four reference PTQ
YAMLs (`EdgeNeXt_{base,small,x_small,xx_small}_config.yaml`). The X5 source
ships **no exporter script and no calibration-data producer**, so this is a
reference configuration set, not a reproducible flow; the gaps are listed
under [Known gaps](#known-gaps). No conversion was executed during this
migration (the OpenExplorer environment was not run).

Unlike most siblings in this family, the EdgeNeXt configs are **positively
anchored**: every YAML's `output_model_file_prefix` reproduces its manifest
basename exactly, and each `onnx_model` name carries its own variant —
compiling a config emits the published filename directly, with no rename
step. `tests/test_conversion_layout.py` pins this agreement.

<a id="source-model"></a>
## Source model

EdgeNeXt base/small/x-small/xx-small (paper [EdgeNeXt: Efficiently
Amalgamated CNN-Transformer Architecture for Mobile Vision
Applications](https://arxiv.org/abs/2206.10589), reference implementation
[mmaaz60/EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt)). The YAMLs expect
`./edgenext_{base,small,x_small,xx_small}.onnx`, but the source delivery
documents no export recipe and pins no weights — the ONNX provenance is
unverified.

<a id="toolchain-targets"></a>
## Toolchain and targets

Model conversion runs on an x86 Linux host inside the RDK X5 OpenExplorer
Docker (march `bayes-e`), never on the board. The source README points at
the generic OE flow (`hb_mapper makertbin`); offline Docker images are
available from the D-Robotics developer forum.

<a id="export"></a>
## Export

No exporter script ships with this delivery. Regenerating the ONNX inputs
requires reproducing the upstream EdgeNeXt export yourself; the resulting
files must be named `edgenext_<variant>.onnx` and placed in this directory
(or the YAMLs' `onnx_model` adjusted). This step is unverified here.

<a id="calibration"></a>
## Calibration

No calibration-data producer ships with this delivery. All four YAMLs
expect `./calibration_data_rgb_f32` (float32 RGB `.npy`) and use
`calibration_type: 'max'` with `max_percentile: 0.999`. Equivalent data
must follow the YAML numerics (mean `123.675 116.28 103.53`, scale
`0.01712475 0.017507 0.01742919`, 224x224); this equivalence is a stated
requirement, not a verified pipeline.

<a id="compile"></a>
## Compile

In the OE environment, with both inputs prepared (base shown; the other
variants substitute their own config):

```bash
# cwd: this conversion directory
# input: ./edgenext_base.onnx + ./calibration_data_rgb_f32
# output: working_dir 'EdgeNeXt_base_224x224_nv12', emitted
#         EdgeNeXt_base_224x224_nv12.bin — equals the manifest name, no rename
hb_mapper makertbin --config EdgeNeXt_base_config.yaml
```

All four YAMLs set `compile_mode: 'latency'` / `optimize_level: 'O3'` and
place their cross-covariance-attention (`xca`) Softmax nodes on the BPU
with int16 I/O via `node_info` — 3 placements per config (stages 1/2/3);
the xx-small model adds 13 further placements (16 total). None carries
`debug_mode` or `set_all_nodes_int16`.

<a id="validation"></a>
## Validation

No x86 reference script ships with this delivery. The functional check is
the unified runtime on board:
`python3 samples/vision/edgenext/runtime/python/main.py --target x5 --asset-id x5:edgenext:EdgeNeXt_base_224x224_nv12.bin ...`
(see [runtime/python/README.md](../runtime/python/README.md)).
**Not run in this migration:** no export, calibration, or compile was
executed; the consistency claims here are static cross-checks of the YAML
contents and filename/prefix agreement with the manifest.

<a id="artifacts"></a>
## Artifacts (kept material)

The four YAML files are kept byte-verbatim from rdk_x5 @ac11571; their
SHA-256 digests are pinned by `tests/test_conversion_layout.py`, so any
future edit is caught by the host suite.

<a id="known-gaps"></a>
## Known gaps

As shipped by the source and preserved here:

1. **No ONNX exporter.** None of the four `edgenext_<variant>.onnx` inputs
   has a producing script, pinned weights, or a documented export recipe.
2. **No calibration-data producer.** `./calibration_data_rgb_f32` has no
   generating script in the source tree.
3. **No pinned compile command.** The source README points at the generic
   OE flow; the exact command that produced each published `.bin` is not
   recorded, so reproduction is unverified.
4. **No conversion executed in this migration.**

The variant-less-prefix rename gap that affects several siblings
(convnext, fasternet, fastvit, efficientnet/efficientformer in B2) does
**not** apply here: the output prefixes carry their variant names.
