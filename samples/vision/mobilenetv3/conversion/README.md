English | [简体中文](README_cn.md)

# MobileNetV3 model conversion

This recipe rebuilds the published MobileNetV3 deployment models from pinned
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
  - `large`: `timm/mobilenetv3_large_100.ra_in1k`, revision
    `96f46a1c52932f27492dff66c72378eb99b443a7`
  - `small`: `timm/mobilenetv3_small_100.lamb_in1k`, revision
    `1824797e7887cbec1990e4adbd6675960a36c589`
- Correspondence: the checkpoints are the upstream timm releases of the
  MobileNetV3-Large-100, MobileNetV3-Small-100 ImageNet-1k classifiers; the weights
  are used unchanged. License: Apache-2.0 (the checkpoint model cards).

<a id="preprocessing"></a>
### Preprocessing contract

Calibration, evaluation, and the runtime share one geometry, which follows the
checkpoint's own evaluation settings (`crop_pct`):

| Variant | `crop_pct` | Shorter edge resized to | Crop |
| --- | --- | --- | --- |
| large | 0.875 | 256 (`int(224 / 0.875)`) | 224x224 center crop |
| small | 0.875 | 256 (`int(224 / 0.875)`) | 224x224 center crop |

The resize is antialiased bicubic (Pillow). Pixels are RGB; the ONNX input
`data` is normalized float32 `[1,3,224,224]` with the ImageNet mean/std and
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
└── ptq_s.yaml  # S100 / S100P / S600 compile configuration (placeholders)
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
| s100 | nash-e | `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0` | hb_compile 3.5.3, HMCT 2.6.5, HBDK 4.7.5 | `ptq_s.yaml` |
| s100p | nash-m | same image as s100 | same as s100 | `ptq_s.yaml` |
| s600 | nash-p | same image as s100 | same as s100 | `ptq_s.yaml` |

There is no X5 build (see [Compile](#compile)).

Documentation: [RDK S toolchain overview](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview),
[D-Robotics toolchain download](https://toolchain.d-robotics.cc/).

<a id="export"></a>
## Export (ONNX)

The commands below build `large`; use `--model v3-small-100` and a separate
work directory for `small`. `WORK` is an absolute directory outside the
repository.

```bash
# cwd: repository root; python 3.10 with the locked requirements installed
export REPO=$(pwd)
export WORK=/abs/path/mobilenetv3-large
python3 utils/tools/mobilenet/workflow.py fetch --model v3-large-100 \
  --output $WORK/weights
python3 samples/vision/mobilenetv3/conversion/export.py --model v3-large-100 \
  --source-dir $WORK/weights \
  --images samples/vision/mobilenetv3/test_data/kit_fox.JPEG \
           samples/vision/mobilenetv3/test_data/zebra_cls.jpg \
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
- Preprocessing: the contract above; the S series takes the normalized ONNX
  input as float32 `.npy`.

```bash
# cwd: repository root; run once per target: s100, s100p, s600
python3 utils/tools/mobilenet/workflow.py calibrate --model v3-large-100 \
  --platform s100 \
  --manifest /abs/calibration/manifest.json \
  --images-root /abs/calibration \
  --expected-images 200 \
  --evaluation-manifest /abs/imagenetv2/manifest.json \
  --output $WORK/s100/calibration
# expect: $WORK/s100/calibration/data (200 files) and calibration.json
```

<a id="compile"></a>
## Compile

Fill the placeholders of the reference configuration (`SIZE` is the model input:
224); the compile parameters (optimization level, latency mode, one
core, NV12 input) are fixed by it:

```bash
# cwd: $WORK
export SIZE=224
for pair in s100:nash-e s100p:nash-m s600:nash-p; do
  target=${pair%%:*}; march=${pair##*:}
  sed -e "s#@WORK@#$WORK#g" -e "s#@TARGET@#$target#g" -e "s#@MARCH@#$march#g" \
      -e "s#@SIZE@#$SIZE#g" \
    $REPO/samples/vision/mobilenetv3/conversion/ptq_s.yaml > $target/config.yaml
done
```

Mount `$WORK` into the container at the same absolute path and run the
toolchain there, for example for S100:

```bash
# cwd: $WORK/s100; the published builds used --gpus all --shm-size 15g
docker run --rm --user "$(id -u):$(id -g)" --gpus all --shm-size 15g \
  -v "$WORK:$WORK" -w "$WORK/s100" \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0 \
  hb_compile --config config.yaml

# S100P and S600: the same with cwd $WORK/<target>
# expect: <target>/build/mobilenetv3_<target>.hbm
```

Rename each output to its published name (see [Artifacts](#artifacts)); the
name carries the march token: `nashe`, `nashm`, `nashp`.

`ptq_s.yaml` sets `quant_config.model_config.all_node_type: int16`: every
activation is INT16 and the weights stay INT8. With INT8 activations the
S builds lose 17.7% (Large) and 99.7% (Small) of the FP32 Top-1; with INT16
activations 0.3% and 1.9-2.2%. There is no X5 build: the X5 toolchain keeps
MobileNetV3's stride-2 depthwise convolutions and SE pooling in INT8 even when
all nodes are set to INT16, and none of its configurations (INT8 with KL, max
or per-channel calibration, bias correction, node-level or full INT16) kept
the Top-1 loss under 5%.

<a id="validation"></a>
## Post-Conversion Validation

1. Inspect the artifact with `hrt_model_exec model_info --model_file <file>`
   on the matching board. The input shapes are per target and variant:

   | Target / variant | Input the metadata exposes | Output |
   | --- | --- | --- |
   | s100/s100p/s600, large | Y `[1,224,224,1]`, UV `[1,112,112,2]` (`mobilenetv3_large_100_nash{e,m,p}_224x224_nv12.hbm`) | F32 `[1,1000]` |
   | s100/s100p/s600, small | Y `[1,224,224,1]`, UV `[1,112,112,2]` (`mobilenetv3_small_100_nash{e,m,p}_224x224_nv12.hbm`) | F32 `[1,1000]` |

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
| `mobilenetv3_large_100_nashe_224x224_nv12.hbm` | s100 | `model/s100/` |
| `mobilenetv3_large_100_nashm_224x224_nv12.hbm` | s100p | `model/s100p/` |
| `mobilenetv3_large_100_nashp_224x224_nv12.hbm` | s600 | `model/s600/` |
| `mobilenetv3_small_100_nashe_224x224_nv12.hbm` | s100 | `model/s100/` |
| `mobilenetv3_small_100_nashm_224x224_nv12.hbm` | s100p | `model/s100p/` |
| `mobilenetv3_small_100_nashp_224x224_nv12.hbm` | s600 | `model/s600/` |

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
