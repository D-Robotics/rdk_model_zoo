English | [简体中文](README_cn.md)

# MobileNetV1 model conversion

This recipe rebuilds the published MobileNetV1 deployment models from pinned
timm checkpoints: FP32 ONNX export, INT8 post-training quantization (PTQ), and
a compile for each RDK target. It runs on an x86 Linux host with Docker. The
export, calibration, and configuration steps use the shared
[host workflow](../../../../utils/tools/mobilenet/README.md); each compile runs
in the matching OpenExplorer (OE) toolchain container.

<a id="source-model"></a>
## Source model

- Framework: PyTorch 2.8.0 with timm 1.0.20 (versions pinned in
  `utils/tools/mobilenet/requirements.lock.txt`).
- Weights, pinned by Hugging Face revision and file SHA-256 in
  `utils/tools/mobilenet/checkpoints.json`:
  - `100`: `timm/mobilenetv1_100.ra4_e3600_r224_in1k`, revision
    `6e88b523abded44469eafcd52f7832d5e0300b1b`
  - `125`: `timm/mobilenetv1_125.ra4_e3600_r224_in1k`, revision
    `9e2b4f9a089f39eaf59cb38fb5d595f4d78c8064`
- Correspondence: the checkpoints are the upstream timm releases of the
  MobileNetV1-100, MobileNetV1-125 ImageNet-1k classifiers; the weights
  are used unchanged. License: Apache-2.0 (the checkpoint model cards).

<a id="preprocessing"></a>
### Preprocessing contract

Calibration, evaluation, and the runtime share one geometry, which follows the
checkpoint's own evaluation settings (`crop_pct`):

| Variant | `crop_pct` | Shorter edge resized to | Crop |
| --- | --- | --- | --- |
| 100 | 0.875 | 256 (`int(224 / 0.875)`) | 224x224 center crop |
| 125 | 0.9 | 248 (`int(224 / 0.9)`) | 224x224 center crop |

The resize is antialiased bicubic (Pillow). Pixels are RGB; the ONNX input
`data` is normalized float32 `[1,3,224,224]` with mean 0.5 / std 0.5 (the
MobileNetV1 checkpoints' own normalization) and
the output `logits` is `[1,1000]`. In the deployed model the normalization is
compiled in (`mean_value` = mean x 255, `scale_value` = 1 / (255 x std)), so the
runtime feeds NV12 produced from the 224x224 crop. NV12 conversion uses
OpenCV's limited-range BT.601 transform, hence `input_space_and_range:
bt601_video` in both configurations.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── export.py  # Pinned timm checkpoint -> verified FP32 ONNX
├── ptq_s.yaml  # S100 / S100P / S600 compile configuration (placeholders)
├── ptq_s_100.yaml  # S configuration of variant 100 (adds weight bias correction)
└── ptq_x5.yaml  # X5 compile configuration (placeholder)
```

Calibration inputs and the OE configuration are produced by
`utils/tools/mobilenet/workflow.py`; `ptq_*.yaml` are the reference compile
configurations of the published artifacts.

<a id="toolchain-targets"></a>
## Toolchain & Targets

Compile with the OE container matching the target and keep its image tag and
digest with the build. Versions used for the published artifacts:

| Target | march | Toolchain image | Tools | Config |
| --- | --- | --- | --- | --- |
| x5 | bayes-e | `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8` | hb_mapper 1.24.3, HBDK 3.49.15, horizon_nn 1.1.0 | `ptq_x5.yaml` |
| s100 | nash-e | `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0` | hb_compile 3.5.3, HMCT 2.6.5, HBDK 4.7.5 | `ptq_s.yaml` (`ptq_s_100.yaml` for 100) |
| s100p | nash-m | same image as s100 | same as s100 | `ptq_s.yaml` (`ptq_s_100.yaml` for 100) |
| s600 | nash-p | same image as s100 | same as s100 | `ptq_s.yaml` (`ptq_s_100.yaml` for 100) |

Documentation: [RDK S toolchain overview](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview),
[D-Robotics toolchain download](https://toolchain.d-robotics.cc/).

<a id="export"></a>
## Export (ONNX)

The commands below build `100`; use `--model v1-125` and a separate
work directory for `125`. `WORK` is an absolute directory outside the
repository.

```bash
# cwd: repository root; python 3.10 with the locked requirements installed
export REPO=$(pwd)
export WORK=/abs/path/mobilenetv1-100
python3 utils/tools/mobilenet/workflow.py fetch --model v1-100 \
  --output $WORK/weights
python3 samples/vision/mobilenetv1/conversion/export.py --model v1-100 \
  --source-dir $WORK/weights \
  --images samples/vision/mobilenetv1/test_data/bulbul.JPEG \
           samples/vision/mobilenetv1/test_data/zebra_cls.jpg \
  --opset 11 --simplify --output $WORK/export
# expect: $WORK/export/model.onnx (data float32 [1,3,224,224] -> logits [1,1000])
#         and export.json with status "passed"
```

`export.py` verifies the checkpoint hashes, loads it strictly, simplifies the
graph, and compares real-image logits and Top-5 with PyTorch. `export.json`
records the maximum absolute error per image and binds the ONNX SHA-256 to
the checkpoint, the preprocessing contract, and the dependency versions.

<a id="calibration"></a>
## Calibration

- Dataset: a fixed 200-image subset of COCO 2017 train2017 (selection seed 42),
  listed with SHA-256 in a manifest. The set shares no image with the
  evaluation set, which `workflow.py calibrate` verifies.
- Preprocessing: the contract above, written per target: X5 takes raw RGB
  float32 in [0,255] (`.rgb`, headerless little-endian; the X5 loader performs
  the RGB/NV12 transformation); the S series takes the normalized ONNX input
  as float32 `.npy`.

```bash
# cwd: repository root; run once per target: x5, s100, s100p, s600
python3 utils/tools/mobilenet/workflow.py calibrate --model v1-100 \
  --platform x5 \
  --manifest /abs/calibration/manifest.json \
  --images-root /abs/calibration \
  --expected-images 200 \
  --evaluation-manifest /abs/imagenetv2/manifest.json \
  --output $WORK/x5/calibration
# expect: $WORK/x5/calibration/data (200 files) and calibration.json
```

<a id="compile"></a>
## Compile

Fill the placeholders of the reference configuration (`SIZE` is the model input:
224); the compile parameters (optimization level, latency mode, one
core, NV12 input) are fixed by it:

```bash
# cwd: $WORK
export SIZE=224
sed -e "s#@WORK@#$WORK#g" -e "s#@SIZE@#$SIZE#g" \
  $REPO/samples/vision/mobilenetv1/conversion/ptq_x5.yaml > x5/config.yaml
for pair in s100:nash-e s100p:nash-m s600:nash-p; do
  target=${pair%%:*}; march=${pair##*:}
  sed -e "s#@WORK@#$WORK#g" -e "s#@TARGET@#$target#g" -e "s#@MARCH@#$march#g" \
      -e "s#@SIZE@#$SIZE#g" \
    $REPO/samples/vision/mobilenetv1/conversion/ptq_s.yaml > $target/config.yaml
done
```

Mount `$WORK` into the container at the same absolute path and run the
toolchain there, for example for X5:

```bash
# cwd: $WORK/x5; the published builds used --shm-size 15g (X5) and,
# for the S images, --gpus all as well
docker run --rm --user "$(id -u):$(id -g)" --shm-size 15g \
  -v "$WORK:$WORK" -w "$WORK/x5" \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8 \
  bash -c "hb_mapper checker --model-type onnx --march bayes-e \
             --model $WORK/export/model.onnx --input-shape data 1x3x${SIZE}x${SIZE} && \
           hb_mapper makertbin --config config.yaml --model-type onnx"

# S100, S100P and S600: cwd $WORK/<target>, image ..._s100_s600_gpu:v3.7.0
#   hb_compile --config config.yaml
# expect: <target>/build/mobilenetv1_<target>.bin (x5) or .hbm (S)
```

Rename each output to its published name (see [Artifacts](#artifacts)); the
name carries the march token: `bayese`, `nashe`, `nashm`, `nashp`.

For variant `100` on S100, S100P and S600 use `ptq_s_100.yaml` instead of
`ptq_s.yaml`: it adds the toolchain's weight bias correction
(`quant_config.model_config.weight.bias_correction`), which keeps the build INT8
and brings its Top-1 loss from 5.3% to 1.3-1.4%. The X5 toolchain's equivalent
(`optimization: bias_correction`) made the X5 build worse, so the published X5
build of `100` uses `ptq_x5.yaml` unchanged.

<a id="validation"></a>
## Post-Conversion Validation

1. Inspect the artifact with `hrt_model_exec model_info --model_file <file>`
   on the matching board. The input shapes are per target and variant:

   | Target / variant | Input the metadata exposes | Output |
   | --- | --- | --- |
   | x5, 100 and 125 | one packed NV12 input, 224x224 (`mobilenetv1_{100,125}_bayese_224x224_nv12.bin`) | F32 `[1,1000,1,1]` |
   | s100/s100p/s600, 100 | Y `[1,224,224,1]`, UV `[1,112,112,2]` (`mobilenetv1_100_nash{e,m,p}_224x224_nv12.hbm`) | F32 `[1,1000]` |
   | s100/s100p/s600, 125 | Y `[1,224,224,1]`, UV `[1,112,112,2]` (`mobilenetv1_125_nash{e,m,p}_224x224_nv12.hbm`) | F32 `[1,1000]` |

   The output is raw logits; the runtime task applies softmax.
2. Smoke test with the sample quick start using the fresh artifact: exit 0 and
   a Top-5 list (see [Expected results](../README.md#expected-results)).
3. Measure the full-set accuracy with the [evaluator](../evaluator/README.md)
   and compare it with the FP32 ONNX baseline of the same checkpoint; the
   published results and the acceptance rule are in the
   [sample README](../README.md#performance).

<a id="artifacts"></a>
## Artifacts

| Artifact | Target | Lands at |
| --- | --- | --- |
| `mobilenetv1_100_bayese_224x224_nv12.bin` | x5 | `model/` |
| `mobilenetv1_100_nashe_224x224_nv12.hbm` | s100 | `model/s100/` |
| `mobilenetv1_100_nashm_224x224_nv12.hbm` | s100p | `model/s100p/` |
| `mobilenetv1_100_nashp_224x224_nv12.hbm` | s600 | `model/s600/` |
| `mobilenetv1_125_bayese_224x224_nv12.bin` | x5 | `model/` |
| `mobilenetv1_125_nashe_224x224_nv12.hbm` | s100 | `model/s100/` |
| `mobilenetv1_125_nashm_224x224_nv12.hbm` | s100p | `model/s100p/` |
| `mobilenetv1_125_nashp_224x224_nv12.hbm` | s600 | `model/s600/` |

The list matches [model/README.md](../model/README.md).

<a id="known-gaps"></a>
## Additional Preparation

- Calibration and evaluation data are external. The COCO train2017 subset and
  the ImageNetV2 MatchedFrequency set (10,000 images) must be obtained from
  their owners; the manifest format and its checks are described in the
  [host workflow](../../../../utils/tools/mobilenet/README.md).
- The OE toolchain images need access to `registry.d-robotics.cc`; log in with
  your own credentials (`docker login`). Credentials never belong in these
  files.
- Compiling for a target needs no board, but validation does: use the board
  that matches the target (S100P and S600 are separate boards from S100).
- The models published before this recipe (different weights, preprocessing and
  a probability output) are no longer distributed with this sample.
