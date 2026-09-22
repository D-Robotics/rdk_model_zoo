# FastViT conversion

This directory is a verbatim rdk_x5 @ac11571 delivery: four reference PTQ
YAMLs (`FastViT_{S12,SA12,T12,T8}_config.yaml`). The X5 source ships **no
exporter script and no calibration-data producer**, so this is a reference
configuration set, not a reproducible flow; the gaps are listed under
[Known gaps](#known-gaps). No conversion was executed during this
migration (the OpenExplorer environment was not run).

Two source quirks are preserved and pinned by
`tests/test_conversion_layout.py` instead of being repaired: every
`onnx_model` points at an **external common model-zoo path** outside this
sample tree, and all four configs share the **variant-less** output prefix
`FastViT_224x224_nv12` (reproducing a published basename needs a rename).

<a id="source-model"></a>
## Source model

FastViT S12/SA12/T12/T8 (paper [FastViT: A Fast Hybrid Vision Transformer
using Structural
Reparameterization](https://arxiv.org/abs/2303.14189), as cited by the
source delivery — no reference implementation link is recorded in the
source). The YAMLs consume their ONNX from the shared `01_common` model
zoo (see gap 1), and the source documents no export recipe and pins no
weights — the ONNX provenance is unverified.

<a id="toolchain-targets"></a>
## Toolchain and targets

Model conversion runs on an x86 Linux host inside the RDK X5 OpenExplorer
Docker (march `bayes-e`), never on the board. The source README points at
the generic OE flow (`hb_mapper makertbin`); offline Docker images are
available from the D-Robotics developer forum.

<a id="export"></a>
## Export

No exporter script ships with this delivery. As shipped, the configs
expect their ONNX inputs at
`../../../01_common/model_zoo/mapper/classification/FastViT/fastvit_<variant>.onnx`
— a path outside this sample that this repository does not carry.
Regenerating an input requires reproducing the upstream FastViT export
yourself and either restoring that layout or adjusting `onnx_model`. This
step is unverified here.

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

In the OE environment, with both inputs prepared (S12 shown; the other
variants substitute their own config):

```bash
# cwd: this conversion directory
# input: the external 01_common ONNX (see gap 1) + ./calibration_data_rgb_f32
# output: working_dir 'FastViT_224x224_nv12_mix', emitted
#         FastViT_224x224_nv12.bin — rename required, see gaps
hb_mapper makertbin --config FastViT_S12_config.yaml
```

All four YAMLs set `compile_mode: 'latency'` / `optimize_level: 'O3'` and
place reparametrized-attention/MLP nodes on the BPU with int16 I/O via
`node_info` (5/6/4/10 placements for S12/SA12/T12/T8). None carries
`debug_mode` or `set_all_nodes_int16`.

<a id="validation"></a>
## Validation

No x86 reference script ships with this delivery. The functional check is
the unified runtime on board:
`python3 samples/vision/fastvit/runtime/python/main.py --target x5 --asset-id x5:fastvit:FastViT_S12_224x224_nv12.bin ...`
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

1. **External ONNX inputs.** Every `onnx_model` points at
   `../../../01_common/model_zoo/mapper/classification/FastViT/...`
   outside this sample tree; that directory is not carried by this
   repository, and no exporter script or pinned weights exist for any
   input.
2. **No calibration-data producer.** `./calibration_data_rgb_f32` has no
   generating script in the source tree.
3. **Variant-less output prefix.** All four YAMLs emit
   `output_model_file_prefix: 'FastViT_224x224_nv12'`, so the compiled
   file is `FastViT_224x224_nv12.bin`, not any manifest name
   (`FastViT_{S12,SA12,T12,T8}_224x224_nv12.bin`). To reproduce a
   published artifact, rename the emitted `.bin` to the manifest name or
   edit the prefix first.
4. **No pinned compile command.** The source README points at the generic
   OE flow; the exact command that produced each published `.bin` is not
   recorded, so reproduction is unverified.
5. **No conversion executed in this migration.**
