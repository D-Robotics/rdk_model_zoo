# YOLOE PF conversion

<a id="source-model"></a>
## Source Model

This directory prepares calibration data and a target-specific OE configuration from a **local, already exported** YOLOE PF ONNX model. It does not download checkpoints. Version 11 exposes DFL16 heads; version 26 exposes direct LTRB heads. Both use the fixed 4585-class vocabulary, three strides (8/16/32), 32 mask coefficients and a prototype tensor. Text/visual-prompt models are incompatible.

Recipes for the three routes — X5 E11, S E11 and S E26 — are provided below. The preparation retains floating output nodes by omitting `remove_node_type`/`remove_node_name` requests. The published S artifacts have quantized outputs and stay separate from a newly converted float model.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── tests/  # Automated tests
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── calibration.py  # Python script
├── configuration.py  # Python script
├── contract.py  # Python script
├── export.py  # Python script
├── export_heads.py  # Python script
├── prepare.py  # Python script
├── requirements-export.txt  # Source or data file
└── requirements-host.txt  # Source or data file
```

<a id="toolchain-targets"></a>
## Toolchain & Targets

| `--target` | `--variant` | March | Compiler | Calibration |
| --- | --- | --- | --- | --- |
| x5 | 11s, 11m, 11l | bayes-e | `hb_mapper makertbin` | raw RGB float32, 0..255 |
| s100 | 11s | nash-e | `hb_compile` | NPY RGB float32, 0..1 |
| s100 | 26n, 26s, 26m, 26l, 26x | nash-e | `hb_compile` | NPY RGB float32, 0..1 |
| s100p | 26n, 26s, 26m, 26l, 26x | nash-m | `hb_compile` | NPY RGB float32, 0..1 |

All inputs are static `[1,3,640,640]` RGB float32 in the ONNX graph. Runtime input is NV12. S600, X5 E26 and S100P E11 are rejected; there is no cross-target fallback. The preparation entry supports all 14 target/variant combinations. The X5 recipe provides an 11s YAML and documents a second attention override required for 11l; applying those policies to real models needs compiler validation. Toolchain versions by route: the S E26 recipe uses the validated OE 3.7.0 CPU container; the S E11 quantized recipe requires D-Robotics OpenExplorer >= 3.0.31 and Ultralytics >= 8.3.0; the float export route pins its dependencies in `requirements-export.txt`.

General OE resources: [OE environment documentation](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview) and [toolchain download](https://toolchain.d-robotics.cc/).

Use Python 3.10+ in a separate environment for host preparation:

```bash
# cwd: repository root
python3 -m pip install -r samples/vision/yoloe/conversion/requirements-host.txt
python3 samples/vision/yoloe/conversion/prepare.py --help
```

These dependencies cover graph inspection, image preparation and YAML only. They do not install OE, PyTorch, Ultralytics or ONNX Runtime. To compile, use the matching X5/S OE environment and install the preparation dependencies there if needed. Do not replace that environment's compiler dependencies indiscriminately.

<a id="export"></a>
## Export

The [export.py](export.py) loads an existing local PF checkpoint, verifies its head family, size and ordered vocabulary, and writes a fresh export directory.

To obtain weights: clone the upstream YOLOE repository (<https://github.com/um-assn/yoloe.git>) with `pip install -r requirements.txt && pip install ultralytics`, and download the PF weights from the Ultralytics assets release, e.g. `wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg-pf.pt` (replace `11s` with `11m`/`11l` for the other E11 sizes). The S E11 quantized route additionally uses the Model Zoo exporter <https://github.com/D-Robotics/rdk_model_zoo/blob/main/demos/Seg/YOLOE-11-Seg-Prompt-Free/YOLOE-11-Seg-Prompt-Free_YUV420SP/cauchy_yoloe11segPF_export.py>, which performs the equivalent module replacements without retraining.

Install the separate export dependencies first (Ultralytics is pinned to the E26 recipe version):

```bash
# cwd: repository root; use a separate CPU export environment
python3 -m pip install -r samples/vision/yoloe/conversion/requirements-export.txt
python3 samples/vision/yoloe/conversion/export.py --help
python3 samples/vision/yoloe/conversion/export.py \
  --weights /work/checkpoints/yoloe-11s-seg-pf.pt --variant 11s \
  --output-dir /work/export11s --test-image samples/vision/yoloe/test_data/office_desk.jpg
python3 samples/vision/yoloe/conversion/export.py \
  --weights /work/checkpoints/yoloe-26n-seg-pf.pt --variant 26n \
  --output-dir /work/export26n --test-image samples/vision/yoloe/test_data/office_desk.jpg
```

`--weights`, `--variant` and `--output-dir` are required. The eight variants are 11s/m/l and 26n/s/m/l/x. `--threads` defaults to 2 and must be positive. If `--test-image` is omitted, validation uses a CPU random tensor with seed 0; an image is recommended for a meaningful model comparison. No checkpoint download or compiler runs implicitly. Existing output directories are rejected. An export failure exits nonzero (2 for handled input/validation errors) and, once preparation starts, preserves `export.json` with the failed stage rather than reporting success.

E11 uses the source cv2/cv3/cv5 branches, DFL16 and opset 11. E26 uses one2one branches, direct LTRB and opset 17. Linear vocabulary weights are applied as equivalent dense 1x1 convolutions without replacing checkpoint parameters; existing vocabulary convolutions are retained. Raw heads return all anchors and ten NHWC tensors, with no proposal filtering. Backbone layer routing and `model.<index>` node names are preserved for X5 attention configuration.

For both families, the exporter compares all raw anchors, decoded pre-Top-K tensors and prototypes with the original static PF head (rtol/atol 1e-4), checks the ONNX graph, then compares all ten ONNX Runtime CPU outputs with PyTorch (rtol/atol 2e-3). The comparison uses `ORT_DISABLE_ALL` to evaluate the exported graph with CPU fusion disabled. It writes `yoloe_<variant>_seg_pf.onnx`, `yoloe_<variant>_seg_pf.names` and `export.json`; the record includes checkpoint/input/ONNX/vocabulary hashes, dependency versions, per-output maximum absolute errors and `status=float_checked`. The export is platform-independent — no hardware march is embedded; the march is selected by the later target-specific preparation and compile steps.

For E26 it additionally requires the exact same selected `(anchor, class)` set as upstream, then compares selected values by that identity. Order changes caused by floating rounding are recorded as `order_identical=false` and a `reordered_rows` count. Any added/dropped anchor or class fails, even when scores are close. The exported interface contains dense tensors. Evaluate final Top-K rankings with the runtime and dataset evaluator.

A reference float-export comparison with the eight checkpoints (E11s/m/l and E26n/s/m/l/x) on the bundled `office_desk.jpg` observed 2/4/2 reordered Top-K rows for E26 m/l/x respectively, with identical selected `(anchor, class)` sets in this one-image CPU comparison. Use the evaluator for dataset accuracy and compiled BIN/HBM inference measurements.

`prepare.py` runs ONNX checker, rejects external tensor files and dynamic shapes, and requires the vocabulary file to match [classes.names](../test_data/classes.names) byte for byte. It matches outputs by unique shape, not physical output order:

| Role | E11 NHWC shape | E26 NHWC shape |
| --- | --- | --- |
| classification, stride `s` | `[1,640/s,640/s,4585]` | same |
| boxes, stride `s` | `[1,640/s,640/s,64]` | `[1,640/s,640/s,4]` |
| mask coefficients, stride `s` | `[1,640/s,640/s,32]` | same |
| prototypes | `[1,160,160,32]` | same |

Here `s` is 8, 16 or 32, for ten float32 outputs total. Matching dimensions certify the tensor interface; checkpoint size, architecture, class semantics and accuracy are certified by the exporter checks and the evaluator. `variant_declared` in the report marks the variant as caller-declared at this stage. Preserve the exporter metadata alongside the preparation directory.

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

The generated command is `hb_compile -c <absolute config.yaml>` on S, or `hb_mapper makertbin --model-type onnx --config <absolute config.yaml>` on X5, with the preparation directory as cwd. The YAML references the ONNX, vocabulary and calibration inputs by absolute path; run the preparation inside the compiler's filesystem/container so every referenced input is accessible there.

The S OE 3.7.0 CPU container validated for the E26 recipe:

```bash
REPO_DIR=/path/to/rdk_model_zoo
docker run --rm -it --shm-size=2g \
  -v "$REPO_DIR":/workspace \
  -w /workspace \
  --entrypoint /bin/bash \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

| Policy | X5 E11 | S E11 | S E26 |
| --- | --- | --- | --- |
| Calibration | default | default, Softmax int8 | KL, all-node int8 |
| Compile | latency, O3, core 1, jobs 4 | latency, O2, core 1, jobs 15, advice 1 | latency, O2, core 1, jobs 4 |
| Padding | pyramid input | input/output no-padding | input no-padding, output padding permitted |
| Output-removal requests | none | none | none |

The X5 int16 attention override is added only for actual ONNX nodes: `/model.10/m/m.0/attn/Softmax` for all E11 sizes and additionally `/model.10/m/m.1/attn/Softmax` for 11l. Each missing expected node is named in a `source attention node absent:...` warning. Do not silently rename a different node into that override. The S E11 quantized recipe `config_ultralytics_YOLOE_Seg_YUV420SP_NV12.yaml` uses NV12 runtime input with `scale_value 0.003921568627451`, default calibration with a softmax-int8 `quant_config`, latency/O2, `jobs: 15`, `advice: 1`, and input/output no-padding; its active `remove_node_name` list uses the v8 head numbering (`/model.23/...`) while YOLOE-11 heads live under `/model.22/...`, so those removal names do not match an 11-series graph. The float route omits removal entirely. [S model modification rules](https://developer.d-robotics.cc/oe_s_doc/guide/model_deployment_guidance/model_deployment_principle_process/model_modify) explain how those options remove boundary operators. Omission expresses float-output intent; final compiler metadata must still verify it.

<a id="validation"></a>
## Validation

Exit codes: 0 for preparation or an exit-zero compiler producing a nonempty artifact, 1 for a captured compiler failure/missing artifact, 2 for invalid inputs or unavailable dependencies/compiler. A successful compile reports **`compiled_unverified`**, with `observed_output_dtype=null`, `board` and `dataset_accuracy` unset. The `_float` filename suffix is an intended contract, not proof of actual precision.

For the published X5 E11 artifacts, validate with the `hb_perf` visualization plus the `hrt_model_exec model_info` inspection (run on a matching X5 image after preparing the BIN):

```bash
hb_perf samples/vision/yoloe/model/x5/yoloe_11s_seg_pf_bayese_640x640_nv12.bin
hrt_model_exec model_info --model_file samples/vision/yoloe/model/x5/yoloe_11s_seg_pf_bayese_640x640_nv12.bin
```

Inspect the compiled model metadata and compare all ten outputs against the float model on representative inputs. The runtime additionally checks target, NV12 input and ten NHWC float32 output roles. Integer output, wrong dimensions or an incompatible target fail; no manual dequantization is inserted into postprocessing. Use the artifact digest from `conversion.json` to identify an explicitly selected local file:

```bash
# cwd: repository root on the matching board; replace the path and digest
python3 samples/vision/yoloe/runtime/python/main.py \
  --target s100 --variant 26n --model-path /work/model.hbm \
  --local-float-sha256 REPLACE_WITH_64_HEX_SHA256
```

Host tests exercise real ONNX validation with synthetic graphs, calibration pixels, all 14 target/variant configurations, and fake compiler success/failure capture. Run them after installing host dependencies:

```bash
# cwd: repository root
python3 -m unittest discover -s samples/vision/yoloe/tests
# Additionally, in the export environment:
python3 -m unittest discover -s samples/vision/yoloe/conversion/tests
```

<a id="artifacts"></a>
## Artifacts

Each preparation directory contains `source/model.onnx`, `source/classes.names`, `calibration/`, `calibration.json`, `config.yaml` and `conversion.json`. Records bind original images, generated tensors, ONNX, vocabulary and YAML with SHA-256. Compilation adds the complete combined stdout/stderr `compile.log`, exact argv/cwd/UTC timestamps/return code and, on success, the artifact path/hash/size in `conversion.json`.

Expected output is `compiler_output/yoloe_<variant>_seg_pf_<march-without-hyphen>_640x640_nv12_float.bin` (X5) or `.hbm` (S). Keep these local conversions separate from the [published artifacts](../model/README.md); no new publication identity or board result is created automatically.

<a id="known-gaps"></a>
## Preparation requirements

Float S HBMs come from the local route above; no float S HBM is published, and the published quantized S artifacts are distinct inputs. Export comparisons run on unoptimized ONNX Runtime CPU; validate optimized-engine behavior, broader inputs and X5 11m/11l compilation with your own runs. Graph/interface validation does not detect an incorrectly labelled checkpoint size or changed label semantics; the exporter checks and evaluator cover those. SDK build and board execution for the [C++ runtime](../runtime/cpp/README.md) follow its guide. The [evaluator](../evaluator/README.md) provides explicit dataset mapping and scoring.

Record the local checkpoint SHA-256, exported ONNX SHA-256 and resulting HBM SHA-256 with each conversion. Keep original quantized-output and float-output routes as distinct artifact records.
