# EfficientNet conversion

This directory merges two source recipes. The S side (rdk_s @380e1a2) is a
complete recipe — exporter scripts, a calibration script, and per-variant
YAMLs whose numerics are mutually consistent (verified during migration).
The X5 side (rdk_x5 @ac11571) ships reference PTQ YAMLs only; its gaps are
listed under [Known gaps](#known-gaps) and are **not** presented as a
reproducible flow. No conversion was executed during this migration (the
OpenExplorer environment was not run); see [Validation](#validation).

<a id="source-model"></a>
## Source model

- S (lite0..lite4): EfficientNet-Lite checkpoints exported through `timm`
  (`tf_efficientnet_lite0.in1k` .. `tf_efficientnet_lite4.in1k`), the
  TensorFlow TPU EfficientNet-Lite family as cited by the source delivery
  (<https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>).
- X5 (b2/b3/b4): EfficientNet B2/B3/B4. The source delivery documents a
  reference timm export flow (create_model → torch.onnx.export →
  onnxsim.simplify) but ships **no exporter script and no pinned
  weights**; the ONNX provenance is therefore unverified.

<a id="toolchain-targets"></a>
## Toolchain and targets

Model conversion runs on an x86 Linux host inside the OpenExplorer Docker
for the target platform, never on the board.

- S100: march `nash-e` (the shipped YAML value).
- S600: the same config with march changed to `nash-p` (edit the YAML or
  pass the toolchain's march override), per the source delivery's note;
  quantization configuration is otherwise identical.
- X5: march `bayes-e`.

- OE resource entry (Docker + development package):
  <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain manual: <https://toolchain.d-robotics.cc/>

<a id="export"></a>
## Export (S recipe)

cwd: this `conversion/` directory. Install the export dependencies first
(`pip install timm onnx onnxsim` inside a suitable Python 3 environment),
then run the matching exporter, for example:

```bash
# input: timm checkpoint tf_efficientnet_lite0.in1k (downloaded by timm)
# output: ./tf_efficientnet_lite0.onnx (opset 11, onnxsim-simplified)
# success: script prints the parameter count and "Simplified model is valid."
python3 get_efficientnet_lite0_onnx.py
```

| Variant | Exporter | ONNX file | Prefix (== manifest basename) | Input |
| --- | --- | --- | --- | --- |
| lite0 | `get_efficientnet_lite0_onnx.py` | `tf_efficientnet_lite0.onnx` | `efficientnet_lite0_224x224_nv12` | 224x224 |
| lite1 | `get_efficientnet_lite1_onnx.py` | `tf_efficientnet_lite1.onnx` | `efficientnet_lite1_240x240_nv12` | 240x240 |
| lite2 | `get_efficientnet_lite2_onnx.py` | `tf_efficientnet_lite2.onnx` | `efficientnet_lite2_260x260_nv12` | 260x260 |
| lite3 | `get_efficientnet_lite3_onnx.py` | `tf_efficientnet_lite3.onnx` | `efficientnet_lite3_300x300_nv12` | 300x300 |
| lite4 | `get_efficientnet_lite4_onnx.py` | `tf_efficientnet_lite4.onnx` | `efficientnet_lite4_380x380_nv12` | 380x380 |

`timm2onnx_local.py` is the source delivery's alternative helper for
exporting from a locally downloaded checkpoint file instead of the timm
hub; edit its `model_name` for the target variant.

<a id="calibration"></a>
## Calibration (S recipe)

cwd: this `conversion/` directory. One script serves all five variants:

```bash
# input: ILSVRC2012_val_*.JPEG images (script uses the first 100 it finds)
# output: ./calibration_data_rgb/*.npy (float32), matching the YAMLs' cal_data_dir
# success: "成功生成 ... 个校准数据文件" (100 files listed)
python3 get_calibration_data.py
```

The preprocessing chain is `ShortSideResize(224) → CenterCrop(224) →
HWC2CHW → Scale(255.0) → Mean([127,127,127]) → Scale(0.007843)`, which was
cross-checked against every YAML's `mean_value: 127 127 127` /
`scale_value: 0.007843 0.007843 0.007843` — the script and the YAMLs are
consistent (unlike the ResNet152 source recipe, whose script and YAML
disagree; that discrepancy does not exist here).

Two source-recipe facts to know before running:

1. The script's default `src_image_dir` points at the legacy rdk_s source
   tree's OpenExplorer calibration directory
   (`../../../open_explorer/samples/ai_toolchain/.../calibration_data/imagenet/`).
   That path does **not** resolve in this repository — the directory does
   not exist here. Edit `src_image_dir` to a directory containing your own
   ILSVRC2012 validation JPEGs before running.
2. The resize/crop is fixed at 224 for **all** variants (including the
   240/260/300/380 lite1..lite4 models); the source recipe reused one
   calibration set for every variant. This is recorded as-is, not
   re-validated.

<a id="compile"></a>
## Compile (S recipe)

cwd: this `conversion/` directory, inside the OE Docker, after export and
calibration. Per the source delivery:

```bash
# S100 build (input: ./tf_efficientnet_lite0.onnx + ./calibration_data_rgb)
# output: ./model_output/efficientnet_lite0_224x224_nv12.hbm
hb_compile --config efficientnet_lite0_config.yaml
```

For S600, change `march` from `nash-e` to `nash-p` in the YAML first (or
pass the toolchain's march override). The output prefix equals the
manifest basename exactly, so the emitted `.hbm` can be copied to
`model/s100/` or `model/s600/` unrenamed. Rebuilding is only needed when
changing the source model or conversion settings — the runtime sample
downloads the published artifacts.

The X5 YAMLs are compiled with the corresponding X5 OE flow
(`hb_mapper`/`hb_compile` with the config file); see
[Known gaps](#known-gaps) item 5 for what is not pinned there.

<a id="validation"></a>
## Validation

- `x86_inference.py` (S recipe) runs ONNX/HBIR/HBM reference inference on
  an x86 host inside the OE environment (it imports `horizon_tc_ui`); use
  it to compare the compiled model against the ONNX float model.
- The functional check on-board is the unified runtime:
  `python3 samples/vision/efficientnet/runtime/python/main.py --target s100 --asset-id s:efficientnet:s100/efficientnet_lite0_224x224_nv12.hbm ...`
  (see [runtime/python/README.md](../runtime/python/README.md)).
- **Not run in this migration:** no export, calibration, compile, or
  x86 comparison was executed (no OE environment was used). The S-recipe
  consistency claims above are static cross-checks of script ↔ YAML
  numerics and filename/prefix agreement with the manifests.

<a id="artifacts"></a>
## Artifacts (kept material)

All 16 files are kept byte-verbatim from the source branches (X5
@ac11571: three YAMLs; S @380e1a2: five YAMLs, five exporters, the
calibration script, `timm2onnx_local.py`, `x86_inference.py`). Their
SHA-256 values are pinned by `tests/test_conversion_layout.py`, so any
future edit to a conversion file is caught by the host suite. Notable
verbatim-kept contents: the S YAMLs use `working_dir: './model_output'`,
`calibration_type: 'max'`, `optimize_level: 'O2'`; the X5 YAMLs use
`calibration_type: 'default'`, `compile_mode: 'latency'`,
`optimize_level: 'O3'`, and B2/B4 carry `node_info` int16 placements
that B3 does not.

<a id="known-gaps"></a>
## Known gaps

The X5 recipe is a reference configuration, not a verified reproducible
flow. As shipped by the source and preserved here:

1. **No ONNX exporter for X5.** `./efficientnet_b2.onnx` /
   `efficientnet_b3.onnx` / `efficientnet_b4.onnx` have no producing
   script and no pinned weights or timm commit; the documented timm flow
   is a reference only.
2. **No calibration-data producer for X5.** The YAMLs expect
   `./calibration_data_rgb_f32` (float32 RGB `.npy`), but no script in
   the source tree produces it. Equivalent data must follow the YAML
   numerics (mean `123.675 116.28 103.53`, scale `0.01712475 0.017507
   0.01742919`, 224x224); this equivalence is a stated requirement, not a
   verified pipeline.
3. **Variant-less output prefix on X5.** All three YAMLs emit
   `output_model_file_prefix: 'EfficientNet_224x224_nv12'`, so the
   compiled file does not carry the B2/B3/B4 identity and building
   another variant into the same working directory collides. B3
   additionally uses `working_dir: 'model_output'` while B2/B4 use
   `'EfficientNet_224x224_nv12'`. To reproduce the manifest artifacts,
   rename the emitted `.bin` to the manifest name
   (`EfficientNet_B2_224x224_nv12.bin`, ...) or edit the prefix per
   variant first.
4. **`debug_mode: 'dump_calibration_data'`** is present in all three X5
   YAMLs (kept verbatim; its effect on the published builds is not
   re-verified here).
5. **No pinned X5 compile command.** The source README points at the
   generic OE flow (`hb_mapper checker` / `hb_mapper makertbin` /
   `hb_compile`); the exact command that produced the published `.bin`
   files is not recorded, so X5 reproduction is unverified.
6. **No conversion executed in this migration** (see
   [Validation](#validation)); in particular the S-side 224-crop
   calibration for 240/260/300/380 models (calibration section, item 2)
   was not re-run or numerically confirmed.
