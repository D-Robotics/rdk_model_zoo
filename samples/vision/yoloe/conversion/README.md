# YOLOE PF conversion

<a id="source-model"></a>
## Source Model

This directory prepares calibration data and an auditable, target-specific OE configuration from a **local, already exported** YOLOE PF ONNX model. It does not download checkpoints. Version 11 exposes DFL16 heads; version 26 exposes direct LTRB heads. Both use the fixed 4585-class vocabulary, three strides (8/16/32), 32 mask coefficients and a prototype tensor. Text/visual-prompt models are incompatible.

The source recipes are preserved in [X5 E11](../../../../platforms/x5/samples/vision/yoloe/conversion/README.md), [S E11](../../../../platforms/s/samples/vision/yoloe11_seg/conversion/README.md), and [S E26](../../../../platforms/s/samples/vision/yoloe26_seg/conversion/README.md). The canonical path retains floating output nodes: it does not request `remove_node_type` or `remove_node_name`. Original S publications have quantized outputs and cannot be substituted for a newly converted float model.

<a id="toolchain-targets"></a>
## Toolchain & Targets

| `--target` | `--variant` | March | Compiler | Calibration |
| --- | --- | --- | --- | --- |
| x5 | 11s, 11m, 11l | bayes-e | `hb_mapper makertbin` | raw RGB float32, 0..255 |
| s100 | 11s | nash-e | `hb_compile` | NPY RGB float32, 0..1 |
| s100 | 26n, 26s, 26m, 26l, 26x | nash-e | `hb_compile` | NPY RGB float32, 0..1 |
| s100p | 26n, 26s, 26m, 26l, 26x | nash-m | `hb_compile` | NPY RGB float32, 0..1 |

All inputs are static `[1,3,640,640]` RGB float32 in the ONNX graph. Runtime input is NV12. S600, X5 E26 and S100P E11 are rejected; there is no cross-target fallback. These 14 selections have host configuration coverage, **not OE compilation acceptance**. The X5 source contains only an 11s YAML; applying that policy to 11m/11l needs real compiler validation. S26's source records OE 3.7.0; a minimum accepted toolchain for the other paths has not been established.

Use Python 3.10+ in a separate environment for host preparation:

```bash
# cwd: repository root
python3 -m pip install -r samples/vision/yoloe/conversion/requirements-host.txt
python3 samples/vision/yoloe/conversion/prepare.py --help
```

These dependencies cover graph inspection, image preparation and YAML only. They do not install OE, PyTorch, Ultralytics or ONNX Runtime. To compile, use the matching X5/S OE environment and install the preparation dependencies there if needed. Do not replace that environment's compiler dependencies indiscriminately.

<a id="export"></a>
## Export

Canonical checkpoint exporters are still being migrated. Until then, the preserved [E11 exporter](../../../../platforms/x5/samples/vision/yoloe/conversion/onnx_export/export_yoloe11seg_bpu.py) and [E26 exporter](../../../../platforms/s/samples/vision/yoloe26_seg/conversion/onnx_export/export_yoloe26_seg_pf.py) are the source implementations. Their separate dependency/version instructions are in the source conversion READMEs above; E26 pins Ultralytics 8.4.127. Put checkpoints in a fresh working directory; these scripts reject existing output paths.

```bash
# cwd: repository root; local PF checkpoint and source export dependencies required
python3 platforms/x5/samples/vision/yoloe/conversion/onnx_export/export_yoloe11seg_bpu.py \
  --weights /work/export11/yoloe-11s-seg-pf.pt --imgsz 640 --opset 11
python3 platforms/s/samples/vision/yoloe26_seg/conversion/onnx_export/export_yoloe26_seg_pf.py \
  --weights /work/checkpoints/yoloe-26n-seg-pf.pt --size n \
  --output-dir /work/export26n
```

E11 writes `.onnx`, `.names` and `.export.json` beside the checkpoint by default, but does not establish accuracy equivalence. E26 writes `yoloe_26n_seg_pf.onnx/.names/.json`, checks the raw head against upstream and compares ONNX Runtime float outputs; its old metadata hard-codes nash-e even though the exported graph is not an HBM. Select the real target explicitly in preparation. Neither exporter has been executed with a real checkpoint during this migration increment.

`prepare.py` runs ONNX checker, rejects external tensor files and dynamic shapes, and requires the vocabulary file to match [classes.names](../test_data/classes.names) byte for byte. It matches outputs by unique shape, not physical output order:

| Role | E11 NHWC shape | E26 NHWC shape |
| --- | --- | --- |
| classification, stride `s` | `[1,640/s,640/s,4585]` | same |
| boxes, stride `s` | `[1,640/s,640/s,64]` | `[1,640/s,640/s,4]` |
| mask coefficients, stride `s` | `[1,640/s,640/s,32]` | same |
| prototypes | `[1,160,160,32]` | same |

Here `s` is 8, 16 or 32, for ten float32 outputs total. Matching dimensions certify the interface only: they do not prove checkpoint size, architecture, class semantics or accuracy. `variant_declared` in the report records this boundary. Preserve the exporter metadata alongside the preparation directory.

<a id="calibration"></a>
## Calibration

Supply representative JPG/JPEG/PNG/BMP images with `--cal-images`. The script recursively sorts relative paths, uses evenly spaced indices and selects up to `--sample-count` (default 100). Fewer than 20 images produce a warning; unreadable selected images fail. Synthetic/unit-test images are not an accuracy calibration set.

Geometry is shared with runtime: E11 uses letterbox with truncated resized dimensions and padding 127; E26 uses rounded resized dimensions and padding 114. Both convert BGR to RGB and NCHW. The output formats differ by toolchain:

- **X5:** `.rgb` contains raw float32 values in 0..255, without a file header. `preprocess_on=False`, `norm_type=data_scale` and `scale_value=1/255` tell Mapper how normalization is incorporated. See [X5 calibration rules](https://developer.d-robotics.cc/oe_x5_doc/cn/oe_mapper/source/ptq/ptq_usage/prepare_calibration_data.html).
- **S:** `.npy` contains float32 values in 0..1, matching the original float model. The runtime NV12 path still needs `scale_value=1/255`; calibration NPY input and runtime image input are different stages, so this is not double normalization. See [S calibration rules](https://developer.d-robotics.cc/oe_s_doc/en/guide/ptq/ptq_usage/prepare_data).

```bash
# cwd: repository root; all /work paths are user-supplied inputs or new outputs
python3 samples/vision/yoloe/conversion/prepare.py \
  --onnx /work/export26n/yoloe_26n_seg_pf.onnx \
  --names /work/export26n/yoloe_26n_seg_pf.names \
  --target s100 --variant 26n --cal-images /work/calibration-images \
  --sample-count 100 --output-dir /work/yoloe26n-s100-config
```

This command performs preparation only and reports `status=config_only`; no compiler runs. For E11, replace ONNX/names with the E11 outputs and select `--variant 11s --target x5` or `s100`. Other supported sizes follow the table. Every invocation needs a new output directory, including a later compile invocation. A failed partial directory is preserved for diagnosis.

<a id="compile"></a>
## Compile

Add `--compile` in the matching OE environment. `--compiler` optionally selects an explicit executable path; it is not a shell command string. For example:

```bash
# cwd: repository root inside the S OE environment
python3 samples/vision/yoloe/conversion/prepare.py \
  --onnx /work/export26n/yoloe_26n_seg_pf.onnx \
  --names /work/export26n/yoloe_26n_seg_pf.names \
  --target s100 --variant 26n --cal-images /work/calibration-images \
  --output-dir /work/yoloe26n-s100-build --compile
```

The generated command is `hb_compile -c <absolute config.yaml>` on S, or `hb_mapper makertbin --model-type onnx --config <absolute config.yaml>` on X5, with the preparation directory as cwd. YAML paths are absolute; prepare inside the compiler's filesystem/container. Moving only the YAML to another filesystem does not relocate its inputs.

| Policy | X5 E11 | S E11 | S E26 |
| --- | --- | --- | --- |
| Calibration | default | default, Softmax int8 | KL, all-node int8 |
| Compile | latency, O3, core 1, jobs 4 | latency, O2, core 1, jobs 15, advice 1 | latency, O2, core 1, jobs 4 |
| Padding | pyramid input | input/output no-padding | input no-padding, output padding permitted |
| Output-removal requests | none | none | none |

The X5 int16 attention override is added only if the actual ONNX contains `/model.10/m/m.0/attn/Softmax`; otherwise `conversion.json` records `source attention node absent`. Do not silently rename a different node into that override. The S11 source YAML contains stale v8 removal names; the canonical float route omits removal entirely. [S model modification rules](https://developer.d-robotics.cc/oe_s_doc/guide/model_deployment_guidance/model_deployment_principle_process/model_modify) explain how those options remove boundary operators. Omission expresses float-output intent; final compiler metadata must still verify it.

<a id="validation"></a>
## Validation

Exit codes: 0 for preparation or an exit-zero compiler producing a nonempty artifact, 1 for a captured compiler failure/missing artifact, 2 for invalid inputs or unavailable dependencies/compiler. A successful compile reports **`compiled_unverified`**, with `observed_output_dtype=null`, `board=not-run` and `dataset_accuracy=not-run`. The `_float` filename suffix is an intended contract, not proof of actual precision.

Before accepting a compiled model, inspect its real metadata and compare all ten outputs against the float model on representative inputs. The canonical runtime additionally checks target, NV12 input and ten NHWC float32 output roles. Integer output, wrong dimensions or an incompatible target fail; no manual dequantization is inserted into postprocessing. Use the artifact digest from `conversion.json` to identify an explicitly selected local file:

```bash
# cwd: repository root on the matching board; replace the path and digest
python3 samples/vision/yoloe/runtime/python/main.py \
  --target s100 --variant 26n --model-path /work/model.hbm \
  --local-float-sha256 REPLACE_WITH_64_HEX_SHA256
```

Host tests exercise real ONNX validation with synthetic graphs, calibration pixels, all 14 target/variant configurations, and fake compiler success/failure capture. They do not run a real model or establish a usable OE recipe. Run them after installing host dependencies:

```bash
# cwd: repository root
python3 -m unittest discover -s samples/vision/yoloe/tests
```

<a id="artifacts"></a>
## Artifacts

Each preparation directory contains `source/model.onnx`, `source/classes.names`, `calibration/`, `calibration.json`, `config.yaml` and `conversion.json`. Records bind original images, generated tensors, ONNX, vocabulary and YAML with SHA-256. Compilation adds the complete combined stdout/stderr `compile.log`, exact argv/cwd/UTC timestamps/return code and, on success, the artifact path/hash/size in `conversion.json`.

Expected output is `compiler_output/yoloe_<variant>_seg_pf_<march-without-hyphen>_640x640_nv12_float.bin` (X5) or `.hbm` (S). Keep these local conversions separate from the [published artifacts](../model/README.md); no new publication identity or board result is created automatically.

<a id="known-gaps"></a>
## Known Gaps

Real checkpoint export, OE compilation, compiled output inspection, dataset accuracy and board inference have not been performed for this canonical conversion. No float S HBM has been published. Source exporters still need canonical integration; X5 11m/11l compile policy and absent-attention warnings need real-model checks. Graph/interface validation cannot detect incorrectly labelled checkpoint size or changed label semantics. C++ migration and a canonical dataset evaluator are separate unfinished work, not implied by this preparation entry.
