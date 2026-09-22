English | [简体中文](./README_cn.md)

# PaddleOCR model conversion

Conversion runs on an x86 Linux host inside the matching RDK OpenExplorer
(OE) environment, never on the board. Two complete, target-separated
recipe families are maintained — choose the one matching the target board:

| Target | Family | Toolchain / march | Committed recipes | Runtime contract they must reproduce |
| --- | --- | --- | --- | --- |
| X5 | PP-OCRv3 English | `hb_mapper`, `bayes-e` (OE X5 v1.2.8) | `x5/ptq_yamls/*.yaml` | packed NV12 detector input; `[1,40,97,1]` recognizer output |
| S100 | PP-OCRv6 | `hb_compile`, `nash-e` (S OpenExplore) | `s100/*_configs.yaml` | split NV12 detector input; `[1,40,18710]` F32 recognizer output; trailing `Dequantize` on the detector |

Graph files, output names, dictionaries, and input protocols are
target-specific. Do not combine the X5 detector with the S100 recognizer or
substitute one dictionary for the other. No S100P/S600 PaddleOCR artifact
row exists, so this directory makes no conversion claim for them.

<a id="source-model"></a>
## Source model

- X5 pair: upstream PaddleOCR PP-OCRv3 English inference models
  (`en_PP-OCRv3_det_infer`, `en_PP-OCRv3_rec_infer`). This repository
  ships **no** PP-OCRv3 exporter; obtain the exported files from the exact
  upstream release and verify their graph inputs before compiling.
- S100 pair: upstream PaddleOCR PP-OCRv6 inference models, exported with
  Paddle2ONNX opset 19 by the command under [export](#export).
- Dictionaries are part of the model contract: the X5 recognizer decodes
  against the fixed 96-character alphabet; the S100 recognizer against
  [`test_data/s100/ppocrv6_dict.txt`](../test_data/s100/ppocrv6_dict.txt)
  (plus blank and trailing space).

<a id="toolchain-targets"></a>
## Toolchain and targets

Use the OE Docker/toolchain matching the target — the audited X5
instructions identify OE X5 v1.2.8, the S instructions the S OpenExplore
toolchain — and record the image tag, OE version, and host date. A generic
load-and-mount sequence:

```bash
# inputs: OE image archive — run the rest inside the container at /workspace
export OE_IMAGE_TAR=/absolute/path/to/oe-image.tar
test -s "$OE_IMAGE_TAR"
docker load -i "$OE_IMAGE_TAR"
docker images
: "${OE_IMAGE:?Set OE_IMAGE to the loaded OE image:tag}"
export REPOSITORY_ROOT="${REPOSITORY_ROOT:-$PWD}"
docker run --rm -it --network host --shm-size=15g \
  -v "$REPOSITORY_ROOT":/workspace --workdir /workspace \
  "$OE_IMAGE" /bin/bash
```

`hb_mapper`, `hb_compile`, and `hrt_model_exec` are supplied by those
environments. The committed S100 recipe uses `nash-e`; `nash-m`/`nash-p`
appear in toolchain documentation for other SoCs but have no audited
artifact or board evidence here — changing `march` is a new conversion
experiment, not a supported variant.

<a id="export"></a>
## Export

S100 PP-OCRv6 ONNX export (upstream PaddlePaddle; network access to clone
and install happens here, on the host, before any OE step):

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
python3 -m pip install -e .
python3 -m pip install paddle2onnx onnxruntime

# outputs: det_onnx/model_detv6.onnx and rec_onnx/model_recv6.onnx
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

Copy the results to the paths the S100 YAMLs expect (or edit only the
documented `onnx_model` field): `conversion/onnx/model_detv6.onnx` and
`conversion/onnx/model_recv6.onnx`.

X5 PP-OCRv3: there is no repository-owned exporter. The recipes expect
files named `en_PP-OCRv3_det_infer.onnx` and `en_PP-OCRv3_rec_infer.onnx`
from the exact PP-OCRv3 release; this sample does not present a guessed
command as a supported X5 export recipe.

<a id="calibration"></a>
## Calibration

The checked-in helper turns a directory of BGR images into the input
tensors the committed recipes expect (no conversion, no network). Run from
`samples/vision/paddle_ocr/conversion`:

```bash
# X5 hb_mapper: two separate empty directories so detector and recognizer
# tensors cannot mix; point the det/rec YAML cal_data_dir at them first.
python3 scripts/prepare_calibration.py \
  --images /data/calibration/images --output ./x5/calibration_data_detector \
  --target x5 --stage detector
python3 scripts/prepare_calibration.py \
  --images /data/recognition/crops --output ./x5/calibration_data_recognizer \
  --target x5 --stage recognizer

# S100 hb_compile: these outputs are the ../calibration_data and
# ../calibration_data_rec_new/cropped_images_npy values in the committed YAMLs.
python3 scripts/prepare_calibration.py \
  --images /data/calibration/images --output ./calibration_data \
  --target s100 --stage detector
python3 scripts/prepare_calibration.py \
  --images /data/recognition/crops \
  --output ./calibration_data_rec_new/cropped_images_npy \
  --target s100 --stage recognizer
```

Detector tensors are resized to 640×640 RGB float32; recognizer tensors to
320×48 RGB float32 in `[0,1]`, matching the runtime's `no_preprocess`
input. The X5 output extension is `.rgbchw`, the S100 extension `.npy`.
Each output directory must be empty, so a second stage cannot silently
reuse residual tensors. Use representative images/crops from the model's
training domain; this helper does not determine an accuracy or a required
sample count.

<a id="compile"></a>
## Compile

X5 PP-OCRv3, inside the OE X5 v1.2.8 environment from `conversion/x5`
after placing the two ONNX files and calibration directories there
(outputs: `model_output/*.bin`; success: `hb_mapper` exits 0):

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

Emitted filenames:

```text
model_output/en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin
model_output/en_PP-OCRv3_rec_infer-deploy_48x320_rgb.bin
```

The detector YAML retains the observed RGB mean/scales, `nv12` runtime
input, `bayes-e`, and O3 compiler level; the recognizer YAML retains
`featuremap` NCHW/no-preprocess and the three observed `p2o.Softmax.*`
int16 node mappings.

S100 PP-OCRv6, inside the S OpenExplore environment from
`conversion/s100` (outputs: `model_output/*.hbm`; success: `hb_compile`
exits 0):

```bash
hb_compile -c paddleocr_det_configs.yaml
hb_compile -c paddleocr_rec_configs.yaml
```

Emitted filenames:

```text
model_output/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
model_output/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm
```

The detector recipe keeps the trailing `Dequantize` node: the audited
runtime reads `fetch_name_0` as an F32 probability map and thresholds it
directly, so removing the node changes the observed contract. The
recognizer keeps the three `p2o.Softmax.*` mappings and
`set_all_nodes_int16`; its class count must stay aligned with the
checked-in dictionary plus blank and trailing space.

<a id="validation"></a>
## Validation

Inspect each generated artifact before board use (record the complete
output):

```bash
hrt_model_exec model_info --model_file \
  model_output/en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin
hrt_model_exec model_info --model_file \
  model_output/en_PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
```

Compare tensor names, shapes, dtypes, and output names with the selected
pair's contract (see [stage I/O](../runtime/python/README.md#stage-io));
register only an artifact whose exact qualified reference and metadata
match. Then confirm on the matching board with the canonical runtime and
the bundled fixture. Status: the recipes are the audited source
configurations carried over verbatim; real OE compilation and board
re-validation of regenerated artifacts are **not-run** in this sample.

<a id="artifacts"></a>
## Artifacts

| Stage | Artifact to retain |
| --- | --- |
| export | the two `.onnx` files, exporter command/version (S100), upstream release identity (X5) |
| calibration | the calibration directories, their image source, and the helper invocation |
| checker | `hb_mapper checker` command and log |
| compile | the YAML used, OE version/image tag, `model_output/` listing |
| deployment | the four manifest filenames listed under [compile](#compile) |
| validation | `hrt_model_exec model_info` output, board run command, predictions |

<a id="known-gaps"></a>
## Known gaps

- No repository-owned PP-OCRv3 (X5) exporter; the upstream export step is
  manual and must be recorded by whoever performs it.
- The audited recipes' original calibration corpora are not checked in;
  the helper prepares tensors but does not fix a corpus or sample count,
  so regenerated artifacts are not claimed equivalent to the published
  ones.
- Real OE compilation and board re-validation of regenerated artifacts:
  **not-run** — only the audited recipes and the host-side helper are
  maintained here.
- `march` values other than the committed ones are unclaimed experiments.
