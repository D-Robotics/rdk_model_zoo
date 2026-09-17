English | [简体中文](./README_cn.md)

# PaddleOCR model conversion

This directory contains the maintained conversion entrypoints for the two
asset families used by the sample. Choose one complete family for the target
board:

| Target | Model family | Toolchain/march | Runtime detector input | Runtime recognizer output | Files |
| --- | --- | --- | --- | --- | --- |
| RDK X5 | PP-OCRv3 English | `hb_mapper`, `bayes-e` | one packed NV12 tensor | `[1,40,97,1]` F32, fixed 96-character alphabet plus blank | `x5/ptq_yamls/*.yaml` |
| RDK S100 | PP-OCRv6 | `hb_compile`, `nash-e` | split `x_y`/`x_uv` NV12 tensors | `[1,40,18710]` F32, checked-in UTF-8 dictionary plus blank/space | `s100/*_configs.yaml` |

The graph files, output names, dictionaries, and input protocols are target
specific. Do not combine the X5 detector with the S100 recognizer or substitute
one dictionary for the other. The repository contains no audited S100P or S600
PaddleOCR artifact row; this directory therefore keeps their conversion status
unclaimed.

## Conversion environment

Run conversion on an x86 Linux host inside the matching OpenExplorer
environment, never on the RDK board. The checked-in X5 instructions identify
OE v1.2.8 and the S-series instructions use the S OpenExplore toolchain. The
toolchain binaries (`hb_mapper`, `hb_compile`, `hrt_model_exec`) are expected
to be supplied by those environments.

For S100 PP-OCRv6 ONNX export, the original source instructions use
Paddle2ONNX opset 19:

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
python3 -m pip install -e .
python3 -m pip install paddle2onnx onnxruntime

paddle2onnx --model_dir ./inference/PP-OCRv6_det_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./inference/det_onnx/model_detv6.onnx \
  --opset_version 19 \
  --enable_onnx_checker True

paddle2onnx --model_dir ./inference/PP-OCRv6_rec_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./inference/rec_onnx/model_recv6.onnx \
  --opset_version 19 \
  --enable_onnx_checker True
```

Copy the resulting files to the paths expected by the S100 YAMLs, or edit only
the documented `onnx_model` field:

```text
conversion/onnx/model_detv6.onnx
conversion/onnx/model_recv6.onnx
```

There is no checked-in PP-OCRv3 exporter in this repository. The X5 recipes
expect upstream/exported files named `en_PP-OCRv3_det_infer.onnx` and
`en_PP-OCRv3_rec_infer.onnx`; obtain those files from the exact PP-OCRv3
release and verify their graph inputs before running `hb_mapper`. The source
audit does not establish a repository-owned exporter command, so this sample
does not present a guessed command as a supported X5 export recipe.

## Calibration data

The helper below turns a directory of BGR images into the input tensors used by
the committed recipes. It performs no model conversion and makes no network
request:

```bash
cd samples/vision/paddle_ocr/conversion

# X5 hb_mapper: use two empty directories so detector and recognizer tensors
# cannot overwrite or mix. Set the detector YAML cal_data_dir to
# './calibration_data_detector' and the recognizer YAML cal_data_dir to
# './calibration_data_recognizer' before compiling.
python3 scripts/prepare_calibration.py \
  --images /data/calibration/images --output ./x5/calibration_data_detector \
  --target x5 --stage detector
python3 scripts/prepare_calibration.py \
  --images /data/recognition/crops --output ./x5/calibration_data_recognizer \
  --target x5 --stage recognizer

# S100 hb_compile: these paths are the ../calibration_data and
# ../calibration_data_rec_new/cropped_images_npy values in the committed YAMLs.
python3 scripts/prepare_calibration.py \
  --images /data/calibration/images --output ./calibration_data \
  --target s100 --stage detector
python3 scripts/prepare_calibration.py \
  --images /data/recognition/crops \
  --output ./calibration_data_rec_new/cropped_images_npy \
  --target s100 --stage recognizer
```

The helper requires each output directory to be empty, so a second stage cannot
silently reuse residual tensors. Choose a new directory for another target or
stage. Detector tensors are resized to 640×640 and stored as RGB float32 values
before the compiler's `data_mean_and_scale` fields. Recognizer tensors are
resized to 320×48 and stored as RGB float32 values in `[0,1]`, matching the
runtime's `no_preprocess` input. The X5 output extension is `.rgbchw`; the S100
output extension is `.npy`. Use representative images/crops from the model's
training domain; this helper does not determine an accuracy or a required
sample count.

## X5 PP-OCRv3 (`hb_mapper`)

Run inside the OE X5 v1.2.8 environment from `conversion/x5` after placing the
two ONNX files and calibration directory there:

```bash
hb_mapper checker --model-type onnx --march bayes-e \
  --model ./en_PP-OCRv3_det_infer.onnx
hb_mapper checker --model-type onnx --march bayes-e \
  --model ./en_PP-OCRv3_rec_infer.onnx

hb_mapper makertbin --model-type onnx \
  --config ptq_yamls/paddleocr_det_config.yaml
hb_mapper makertbin --model-type onnx \
  --config ptq_yamls/paddleocr_rec_config.yaml
```

The YAMLs emit:

```text
model_output/en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin
model_output/en_PP-OCRv3_rec_infer-deploy_48x320_rgb.bin
```

Check the generated metadata before copying artifacts to the X5 board:

```bash
hrt_model_exec model_info --model_file \
  model_output/en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin
hrt_model_exec model_info --model_file \
  model_output/en_PP-OCRv3_rec_infer-deploy_48x320_rgb.bin
```

The detector YAML retains the observed RGB mean/scales, `nv12` runtime input,
`bayes-e`, and O3 compiler level. The recognizer YAML retains `featuremap`
NCHW/no-preprocess and the three observed `p2o.Softmax.*` int16 node mappings.

## S100 PP-OCRv6 (`hb_compile`)

Run inside the S OpenExplore environment from `conversion/s100`:

```bash
hb_compile -c paddleocr_det_configs.yaml
hb_compile -c paddleocr_rec_configs.yaml
```

The YAMLs emit:

```text
model_output/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
model_output/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm
```

The detector recipe keeps the trailing `Dequantize` node. The audited runtime
expects `fetch_name_0` as an F32 probability map and thresholds that value
directly; removing the node changes this observed contract. The recognizer
keeps the three `p2o.Softmax.*` mappings and `set_all_nodes_int16`; its output
class count must remain aligned with
`test_data/s100/ppocrv6_dict.txt` plus the blank and trailing-space classes.
Inspect each generated artifact before board execution:

```bash
hrt_model_exec model_info --model_file \
  model_output/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
hrt_model_exec model_info --model_file \
  model_output/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm
```

The committed S100 recipe uses `nash-e`. `nash-m` and `nash-p` names appear in
toolchain documentation for other SoCs, but this repository has no corresponding
audited manifest artifacts or board evidence; changing `march` is therefore a
new conversion experiment rather than a supported sample variant.

## Before runtime use

Compare model-info tensor names, shapes, dtypes, and output names with the
selected pair in the parent README. Register only the artifact whose exact
qualified manifest reference and metadata match. The Python entrypoint accepts
the artifacts via `--det-model-path`/`--rec-model-path` plus both qualified
asset references; it never infers a target from a filename.
