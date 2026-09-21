# EfficientViT conversion

This directory is a verbatim rdk_x5 @ac11571 delivery: the single reference
PTQ YAML (`EfficientViT_MSRA_m5_config.yaml`). The X5 source ships **no
exporter script and no calibration-data producer**, so this is a reference
configuration, not a reproducible flow; the gaps are listed under
[Known gaps](#known-gaps). No conversion was executed during this migration
(the OpenExplorer environment was not run).

<a id="source-model"></a>
## Source model

EfficientViT-MSRA m5 (paper [EfficientViT: Memory Efficient Vision
Transformer with Cascaded Group
Attention](https://arxiv.org/abs/2305.07027), reference implementation
[microsoft/Cream/EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT)).
The YAML expects `./efficientvit_m5.onnx`, but the source delivery documents
no export recipe and pins no weights — the ONNX provenance is unverified.

<a id="toolchain-targets"></a>
## Toolchain and targets

Model conversion runs on an x86 Linux host inside the RDK X5 OpenExplorer
Docker (march `bayes-e`), never on the board. The source README points at
the generic OE flow (`hb_mapper makertbin`); offline Docker images are
available from the D-Robotics developer forum.

<a id="export"></a>
## Export

No exporter script ships with this delivery. Regenerating the ONNX input
requires reproducing the upstream MSRA EfficientViT export yourself; the
resulting file must be named `efficientvit_m5.onnx` and placed in this
directory (or the YAML's `onnx_model` adjusted). This step is unverified
here.

<a id="calibration"></a>
## Calibration

No calibration-data producer ships with this delivery. The YAML expects
`./calibration_data_rgb_f32` (float32 RGB `.npy`) and uses
`calibration_type: 'max'` with `max_percentile: 0.99999` — the most
conservative percentile in the X5 classification family (the siblings use
0.999-0.9995). Equivalent data must follow the YAML numerics (mean
`123.675 116.28 103.53`, scale `0.01712475 0.017507 0.01742919`, 224x224);
this equivalence is a stated requirement, not a verified pipeline.

<a id="compile"></a>
## Compile

In the OE environment, with both inputs prepared:

```bash
# cwd: this conversion directory
# input: ./efficientvit_m5.onnx + ./calibration_data_rgb_f32
# output: working_dir 'EfficientViT_msra_224x224_nv12', emitted
#         EfficientViT_msra_224x224_nv12.bin — rename required, see gaps
hb_mapper makertbin --config EfficientViT_MSRA_m5_config.yaml
```

The YAML sets `compile_mode: 'latency'` / `optimize_level: 'O3'` and places
28 attention `Softmax` nodes on the BPU with int16 I/O via `node_info` (the
cascaded-group-attention structure). Unlike the EfficientFormerV2 delivery,
it carries no `debug_mode` and no `set_all_nodes_int16` optimization.

<a id="validation"></a>
## Validation

No x86 reference script ships with this delivery. The functional check is
the unified runtime on board:
`python3 samples/vision/efficientvit/runtime/python/main.py --target x5 --asset-id x5:efficientvit:EfficientViT_m5_224x224_nv12.bin ...`
(see [runtime/python/README.md](../runtime/python/README.md)).
**Not run in this migration:** no export, calibration, or compile was
executed; the consistency claims here are static cross-checks of the YAML
contents and filename/prefix agreement with the manifest.

<a id="artifacts"></a>
## Artifacts (kept material)

The YAML file is kept byte-verbatim from rdk_x5 @ac11571; its SHA-256 is
pinned by `tests/test_conversion_layout.py`, so any future edit is caught
by the host suite.

<a id="known-gaps"></a>
## Known gaps

As shipped by the source and preserved here:

1. **No ONNX exporter.** `efficientvit_m5.onnx` has no producing script,
   pinned weights, or a documented export recipe.
2. **No calibration-data producer.** `./calibration_data_rgb_f32` has no
   generating script in the source tree.
3. **Variant-less output prefix.** The YAML emits
   `output_model_file_prefix: 'EfficientViT_msra_224x224_nv12'`, so the
   compiled file is `EfficientViT_msra_224x224_nv12.bin`, not the manifest
   name `EfficientViT_m5_224x224_nv12.bin`. To reproduce the published
   artifact, rename the emitted `.bin` to the manifest name or edit the
   prefix first. (The ONNX input name does carry the m5 identity; only the
   output side is variant-less.)
4. **No pinned compile command.** The source README points at the generic
   OE flow; the exact command that produced the published `.bin` is not
   recorded, so reproduction is unverified.
5. **No conversion executed in this migration.**
