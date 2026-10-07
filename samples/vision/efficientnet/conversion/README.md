# EfficientNet conversion

Use the S recipe below with its exporter scripts, calibration script, and
variant-specific YAMLs. For X5 B2/B3/B4, prepare the matching ONNX graph
and calibration inputs described under [Additional preparation](#known-gaps),
then compile with the corresponding X5 YAML.

<a id="source-model"></a>
## Source model

- S (lite0..lite4): EfficientNet-Lite checkpoints exported through `timm`
  (`tf_efficientnet_lite0.in1k`.. `tf_efficientnet_lite4.in1k`), the
  TensorFlow TPU EfficientNet-Lite family
  (<https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>).
- X5 (b2/b3/b4): EfficientNet B2/B3/B4. Export the selected timm variant
  with `create_model → torch.onnx.export → onnxsim.simplify`, and record
  the checkpoint revision used for the ONNX graph.

<a id="toolchain-targets"></a>
## Toolchain and targets

Run model conversion on an x86 Linux host inside the OpenExplorer Docker
for the target platform.

- S100: march `nash-e` (the shipped YAML value).
- S600: the same config with march changed to `nash-p` (edit the YAML or
  pass the toolchain's march override); quantization configuration is
  otherwise identical.
- X5: march `bayes-e`.

- OE resource entry (Docker + development package):
  <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain manual: <https://toolchain.d-robotics.cc/>


Toolchain resources:

- [OE Docker environment](https://forum.d-robotics.cc/t/topic/35229)

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

`timm2onnx_local.py` is an alternative helper for exporting from a
locally downloaded checkpoint file instead of the timm hub; edit its
`model_name` for the target variant.

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

1. The script's default `src_image_dir` points at the original
   OpenExplorer calibration directory
   (`../../../open_explorer/samples/ai_toolchain/.../calibration_data/imagenet/`),
   which does not exist in this repository. Edit `src_image_dir` to a
   directory containing your own ILSVRC2012 validation JPEGs before
   running.
2. The resize/crop is fixed at 224 for **all** variants (including the
   240/260/300/380 lite1..lite4 models); the recipe reuses one
   calibration set for every variant, as delivered.

<a id="compile"></a>
## Compile (S recipe)

cwd: this `conversion/` directory, inside the OE Docker, after export and
calibration:

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

Compile the X5 YAMLs with the corresponding X5 OE flow
(`hb_mapper`/`hb_compile` with the config file); see
[Additional preparation](#known-gaps) for the required graph and calibration inputs.

<a id="validation"></a>
## Validation

- `x86_inference.py` (S recipe) runs ONNX/HBIR/HBM reference inference on
  an x86 host inside the OE environment (it imports `horizon_tc_ui`); use
  it to compare the compiled model against the ONNX float model.
- The functional check on-board is the unified runtime:
  `python3 samples/vision/efficientnet/runtime/python/main.py --target s100 --asset-id s:efficientnet:s100/efficientnet_lite0_224x224_nv12.hbm...`
  (see [runtime/python/README.md](../runtime/python/README.md)).

<a id="artifacts"></a>
## Artifacts

The 16 recipe files comprise three X5 YAMLs, five S YAMLs, five variant-specific
export scripts, the calibration script, `timm2onnx_local.py` and
`x86_inference.py`. The S YAMLs use
`working_dir: './model_output'`,
`calibration_type: 'max'`, `optimize_level: 'O2'`; the X5 YAMLs use
`calibration_type: 'default'`, `compile_mode: 'latency'`,
`optimize_level: 'O3'`, and B2/B4 carry `node_info` int16 placements
that B3 does not.

<a id="known-gaps"></a>
## Additional preparation

For X5, prepare B2/B3/B4 ONNX graphs and `./calibration_data_rgb_f32` float32 RGB `.npy` inputs. Apply the YAML values: mean `123.675 116.28 103.53`, scale `0.01712475 0.017507 0.01742919`, input 224x224. All three X5 YAMLs use `output_model_file_prefix: 'EfficientNet_224x224_nv12'` and `debug_mode: 'dump_calibration_data'`; B3 uses `working_dir: 'model_output'`, while B2/B4 use `'EfficientNet_224x224_nv12'`. Build each variant in an isolated directory and name its output with the matching manifest filename. Use the checker/compiler commands above and the X5 OE toolchain. For S100/S600, follow the complete `hb_compile` recipes earlier in this guide.
