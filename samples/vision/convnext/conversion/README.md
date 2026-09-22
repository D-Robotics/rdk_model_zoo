# ConvNeXt conversion

This directory is a verbatim rdk_x5 @ac11571 delivery: three reference PTQ
YAMLs (`ConvNeXt_atto.yaml`, `ConvNeXt_femto.yaml`, `ConvNeXt_nano.yaml`).
Only **atto** has a published manifest asset; femto and nano are recipes
without published artifacts. The X5 source ships **no exporter script and
no calibration-data producer**, and the three YAMLs' ONNX references are
mutually inconsistent as shipped — this is a reference configuration set,
not a reproducible flow; the gaps are listed under
[Known gaps](#known-gaps). No conversion was executed during this migration
(the OpenExplorer environment was not run).

<a id="source-model"></a>
## Source model

ConvNeXt (paper [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545),
reference implementation
[facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)).
The atto/femto/nano sizes come from the official ConvNeXt size ladder. The
YAMLs' `onnx_model` entries do **not** line up with their own variant names
(see gap 1); the ONNX provenance of every config is unverified.

<a id="toolchain-targets"></a>
## Toolchain and targets

Model conversion runs on an x86 Linux host inside the RDK X5 OpenExplorer
Docker (march `bayes-e`), never on the board. The source README points at
the generic OE flow (`hb_mapper makertbin`); offline Docker images are
available from the D-Robotics developer forum.

<a id="export"></a>
## Export

No exporter script ships with this delivery. Regenerating an ONNX input
requires reproducing the upstream ConvNeXt export yourself. Note that the
three YAMLs point at **different and cross-swapped** files — as shipped,
the atto config consumes `./convnext_femto.onnx`, the femto config points
at `../../../01_common/model_zoo/mapper/classification/ConvNeXt/convnext_atto.onnx`
(a path outside this sample), and the nano config points at
`./convnext_pico.onnx` (a size not otherwise present in this delivery).
Which file actually produced the published atto artifact is not recorded;
this step is unverified here.

<a id="calibration"></a>
## Calibration

No calibration-data producer ships with this delivery. All three YAMLs
expect `./calibration_data_rgb_f32` (float32 RGB `.npy`) and use
`calibration_type: 'default'`. Equivalent data must follow the YAML
numerics (mean `123.675 116.28 103.53`, scale `0.01712475 0.017507
0.01742919`, 224x224); this equivalence is a stated requirement, not a
verified pipeline.

<a id="compile"></a>
## Compile

In the OE environment, with both inputs prepared (atto shown):

```bash
# cwd: this conversion directory
# input: the YAML's onnx_model target (see gap 1 — as shipped this is
#        ./convnext_femto.onnx for the atto config) + ./calibration_data_rgb_f32
# output: working_dir 'ConvNeXt-deploy_224x224_nv12', emitted
#         ConvNeXt-deploy_224x224_nv12.bin — rename required, see gaps
hb_mapper makertbin --config ConvNeXt_atto.yaml
```

All three YAMLs set `compile_mode: 'latency'` / `optimize_level: 'O3'` and
place depthwise/normalization nodes on the BPU with int16 I/O via
`node_info` (8 placements in atto, 5 in femto, 8 in nano; no Softmax
placements — ConvNeXt has no attention softmax). None carries `debug_mode`
or `set_all_nodes_int16`.

<a id="validation"></a>
## Validation

No x86 reference script ships with this delivery. The functional check is
the unified runtime on board:
`python3 samples/vision/convnext/runtime/python/main.py --target x5 --asset-id x5:convnext:ConvNeXt_atto_224x224_nv12.bin ...`
(see [runtime/python/README.md](../runtime/python/README.md)).
**Not run in this migration:** no export, calibration, or compile was
executed; the consistency claims here are static cross-checks of the YAML
contents and filename/prefix agreement with the manifest.

<a id="artifacts"></a>
## Artifacts (kept material)

The three YAML files are kept byte-verbatim from rdk_x5 @ac11571; their
SHA-256 digests are pinned by `tests/test_conversion_layout.py`, so any
future edit is caught by the host suite.

<a id="known-gaps"></a>
## Known gaps

As shipped by the source and preserved here:

1. **Mutually inconsistent ONNX references.** atto→`./convnext_femto.onnx`,
   femto→`../../../01_common/model_zoo/mapper/classification/ConvNeXt/convnext_atto.onnx`
   (outside the sample tree), nano→`./convnext_pico.onnx`. The source
   README's directory listing also mentions a `ConvNeXt_pico.yaml` that is
   **not shipped**. None of these references is repaired here; no file has
   a producing script, pinned weights, or a documented export recipe.
2. **No calibration-data producer.** `./calibration_data_rgb_f32` has no
   generating script in the source tree.
3. **Variant-less output prefix.** All three YAMLs emit
   `output_model_file_prefix: 'ConvNeXt-deploy_224x224_nv12'`, so the
   compiled file is `ConvNeXt-deploy_224x224_nv12.bin`, not the manifest
   name `ConvNeXt_atto_224x224_nv12.bin`. To reproduce the published
   artifact, rename the emitted `.bin` to the manifest name or edit the
   prefix first.
4. **femto/nano publish no asset.** Only atto has a manifest row; the two
   extra recipes cannot be validated against a published artifact and are
   kept as source material only.
5. **No pinned compile command.** The source README points at the generic
   OE flow; the exact command that produced the published `.bin` is not
   recorded, so reproduction is unverified.
6. **No conversion executed in this migration.**
