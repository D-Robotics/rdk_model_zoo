# FasterNet conversion

This directory is a verbatim rdk_x5 @ac11571 delivery: four reference PTQ
YAMLs (`FasterNet_{S,T0,T1,T2}_config.yaml`). The X5 source ships **no
exporter script and no calibration-data producer**, so this is a reference
configuration set, not a reproducible flow; the gaps are listed under
[Known gaps](#known-gaps). No conversion was executed during this
migration (the OpenExplorer environment was not run).

Two source quirks are preserved and pinned by
`tests/test_conversion_layout.py` instead of being repaired: all four
configs share the **variant-less** output prefix
`FasterNet_224x224_nv12`, and the `working_dir` values are asymmetric
(S uses `model_output`, T0 appends `_mix`, T1/T2 use the plain prefix).

<a id="source-model"></a>
## Source model

FasterNet S/T0/T1/T2 (paper [Run, Don't Walk: Chasing Higher FLOPS for
Faster Neural Networks](https://arxiv.org/abs/2303.03667), as cited by the
source delivery — no reference implementation link is recorded in the
source). The YAMLs expect `./fasternet_{s,t0,t1,t2}.onnx`, but the source
delivery documents no export recipe and pins no weights — the ONNX
provenance is unverified.

<a id="toolchain-targets"></a>
## Toolchain and targets

Model conversion runs on an x86 Linux host inside the RDK X5 OpenExplorer
Docker (march `bayes-e`), never on the board. The source README points at
the generic OE flow (`hb_mapper makertbin`); offline Docker images are
available from the D-Robotics developer forum.

<a id="export"></a>
## Export

No exporter script ships with this delivery. Regenerating the ONNX inputs
requires reproducing the upstream FasterNet export yourself; the resulting
files must be named `fasternet_<variant>.onnx` (lowercase, as the YAMLs
expect) and placed in this directory (or the YAMLs' `onnx_model`
adjusted). This step is unverified here.

<a id="calibration"></a>
## Calibration

No calibration-data producer ships with this delivery. All four YAMLs
expect `./calibration_data_rgb_f32` (float32 RGB `.npy`) and use
`calibration_type: 'default'`. Equivalent data must follow the YAML
numerics (mean `123.675 116.28 103.53`, scale `0.01712475 0.017507
0.01742919`, 224x224); this equivalence is a stated requirement, not a
verified pipeline.

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
hb_mapper makertbin --config FasterNet_S_config.yaml
```

All four YAMLs set `compile_mode: 'latency'` / `optimize_level: 'O3'`.
Only the T0 config places nodes on the BPU with int16 I/O via `node_info`
(2 partial-conv-related placements); S/T1/T2 carry no placements. None
carries `debug_mode` or `set_all_nodes_int16`.

<a id="validation"></a>
## Validation

No x86 reference script ships with this delivery. The functional check is
the unified runtime on board:
`python3 samples/vision/fasternet/runtime/python/main.py --target x5 --asset-id x5:fasternet:FasterNet_S_224x224_nv12.bin ...`
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

1. **No ONNX exporter.** None of the four `fasternet_<variant>.onnx`
   inputs has a producing script, pinned weights, or a documented export
   recipe.
2. **No calibration-data producer.** `./calibration_data_rgb_f32` has no
   generating script in the source tree.
3. **Variant-less output prefix.** All four YAMLs emit
   `output_model_file_prefix: 'FasterNet_224x224_nv12'`, so the compiled
   file is `FasterNet_224x224_nv12.bin`, not any manifest name
   (`FasterNet_{S,T0,T1,T2}_224x224_nv12.bin`). To reproduce a published
   artifact, rename the emitted `.bin` to the manifest name or edit the
   prefix first. (The ONNX input names do carry the variant, lowercase;
   only the output side is variant-less.)
4. **working_dir asymmetry.** S compiles into `model_output`, T0 into
   `FasterNet_224x224_nv12_mix`, T1/T2 into `FasterNet_224x224_nv12` —
   three different conventions in one delivery, preserved verbatim.
5. **No pinned compile command.** The source README points at the generic
   OE flow; the exact command that produced each published `.bin` is not
   recorded, so reproduction is unverified.
6. **No conversion executed in this migration.**
