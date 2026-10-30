English | [简体中文](README_cn.md)

# MobileNetV3 image classification

MobileNetV3 combines inverted residual blocks and squeeze-and-excitation for mobile image classification.

Sources: [timm/models/mobilenetv3.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv3.py) · [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

<a id="overview"></a>

## Overview

From `runtime/python`, run `python3 main.py`; the classifier is `classify.py`.
Boards and artifacts are in the support table below. Options and model selection: [runtime/python/README.md](runtime/python/README.md).

### Algorithm background

MobileNetV3 is the NAS-searched member of the MobileNet family:
architecture search plus NetAdapt tune the block configuration,
squeeze-and-excitation attention recalibrates channel weights inside the
inverted residual blocks, and the h-swish activation keeps the
non-linearity cheap on mobile hardware
([paper](https://arxiv.org/abs/1905.02244),
[timm/models/mobilenetv3.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv3.py)).

Feature summary:

- **Depthwise separable convolution**: retains the efficient MobileNet convolution structure.
- **Inverted residual blocks**: expansion–depthwise–projection blocks for efficient feature extraction.
- **SE attention module**: recalibrates channel weights to improve feature representation.
- **H-Swish activation**: hardware-friendly activation for embedded deployment.
- **Model variants**: this sample ships the MobileNetV3-Large-100 and MobileNetV3-Small-100 deployment models (timm checkpoints; INT16 activations with INT8 weights, S100/S100P/S600 only).

![MobileNetV3 block](./test_data/MobileNetV3_architecture.png)

*MobileNetV3 block (Figure 4 of the paper): the inverted residual block with
squeeze-and-excite applied on the residual path — after the NL depthwise
3×3, a global pool plus FC-ReLU / FC-hard-sigmoid gate modulates the
expanded channels, and the gated result passes through the final NL 1×1
projection (the non-linearity is chosen per layer).*

<a id="directory"></a>
## Directory structure

```text
mobilenetv3/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python inference
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── requirements-host.txt  # Python dependencies
```

<a id="support-matrix"></a>
## Support matrix

The table lists the models published for this sample; `—` marks an
unpublished combination.

| Model | Input | X5 | S100 | S100P | S600 |
| --- | --- | --- | --- | --- | --- |
| MobileNetV3-Large | 224x224 | — | ✓ | ✓ | ✓ |
| MobileNetV3-Small | 224x224 | — | ✓ | ✓ | ✓ |

Runtime coverage: the Python CLI serves every published combination with
`--target auto|s100|s100p|s600` and `--variant large|small` (default
`large`). The S series publishes
`mobilenetv3_<block>_100_nashe/nashm/nashp_224x224_nv12.hbm` under `s100/`,
`s100p/` and `s600/`. No X5 artifact is published, so `--target x5` resolves
no asset. The sample ships no C++ runtime.

<a id="prerequisites"></a>
## Prerequisites

Board Python execution needs the board image's matching `hbm_runtime`,
NumPy, OpenCV-Python, and Pillow (the published models expect an antialiased
bicubic shorter-edge resize, which Pillow performs); the SDK is imported only
when a model executes
(`--help`, `--list-models`, `--dry-run`, and host tests need no SDK).
Manifest reading also needs PyYAML. On a development host, install the
user-space dependencies from `requirements-host.txt`:

```bash
# cwd: repository root — success: "host dependencies: ok"
python3 -m venv .venv-mobilenetv3
source .venv-mobilenetv3/bin/activate
python3 -m pip install -r samples/vision/mobilenetv3/requirements-host.txt
python3 -c "import cv2, numpy, yaml, PIL; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

One complete path on an S100 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row s:mobilenetv3:s100/mobilenetv3_large_100_nashe_224x224_nv12.hbm)
#    output: samples/vision/mobilenetv3/model/s100/mobilenetv3_large_100_nashe_224x224_nv12.hbm
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/mobilenetv3/model/download.sh s100 large

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/mobilenetv3/runtime/python/main.py \
  --target s100 \
  --asset-id s:mobilenetv3:s100/mobilenetv3_large_100_nashe_224x224_nv12.hbm \
  --model-path samples/vision/mobilenetv3/model/s100/mobilenetv3_large_100_nashe_224x224_nv12.hbm \
  --test-img samples/vision/mobilenetv3/test_data/kit_fox.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

For S100P and S600 use the matching `s:` reference (see `--list-models`) and the
same root `datasets/imagenet/` labels. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; Pass `--img-save-path` to save a visualization; otherwise results are printed to stdout. With the bundled `kit_fox.JPEG` the Top-1 is class 278 (`kit fox`)
and with `zebra_cls.jpg` it is class 340 (`zebra`), for both variants on S100, S100P
and S600. Select a target and variant listed in the [Support matrix](#support-matrix), prepare that exact manifest artifact with the model downloader, and run the sample on the matching board.

<a id="performance"></a>
## Performance data

All numbers were measured on real boards with the published artifacts
(INT16 activations with INT8 weights, 224x224, batch 1). The models are 5.48 M parameters / 0.43 GFLOPs (Large-100) and 2.54 M parameters / 0.11 GFLOPs (Small-100);
GFLOPs counts Conv and Gemm multiply-accumulates as two operations.

**Accuracy.** Top-1 / Top-5 over the complete ImageNetV2 MatchedFrequency set
(10,000 images, 1,000 classes). This is not the ILSVRC2012 validation set, so
the values are not comparable with ImageNet-1k validation figures. "FP32" is
the ONNX export of the same checkpoint evaluated on the same crops; "Board" is
the compiled model on the matching board.

| Model | Target | FP32 Top-1 | Board Top-1 | FP32 Top-5 | Board Top-5 |
| --- | --- | --- | --- | --- | --- |
| MobileNetV3-Large-100 | S100 | 63.20% | 62.99% | 83.76% | 83.75% |
| MobileNetV3-Large-100 | S100P | 63.20% | 62.99% | 83.76% | 83.75% |
| MobileNetV3-Large-100 | S600 | 63.20% | 63.00% | 83.76% | 83.74% |
| MobileNetV3-Small-100 | S100 | 54.47% | 53.29% | 77.49% | 76.50% |
| MobileNetV3-Small-100 | S100P | 54.47% | 53.29% | 77.49% | 76.50% |
| MobileNetV3-Small-100 | S600 | 54.47% | 53.44% | 77.49% | 76.65% |

MobileNetV3 is published for S100, S100P and S600 only, with INT16 activations
and INT8 weights. Its hard-swish and squeeze-and-excitation activations differ
by up to about 100x between channels, so INT8 activations lose 18% (Large) to
nearly all (Small) of the FP32 Top-1. The X5 toolchain cannot run MobileNetV3's
stride-2 depthwise convolutions and SE pooling in INT16, and no X5 configuration
kept the loss under 5%, so there is no X5 build. INT16 makes the model runtime
1.5-2.1x slower than an INT8 build of the same model.

**Speed.** Runtime numbers come from `hrt_model_exec perf` on BPU core 0 (the
model only: no preprocessing or postprocessing); FPS is the total completed
frames divided by the common wall time of 3 runs x 200 frames after a 20-frame
warmup. The C++ pipeline column times one frame from an in-memory BGR image to
a Top-5 list, including resize, crop, NV12 packing, input upload, inference,
and ranking (file reading and decoding excluded), for 1 and 2 independent
streams.

| Model | Target | Runtime latency, 1 thread (ms) | Runtime FPS, 1 / 2 threads | C++ pipeline FPS, 1 / 2 streams | CPU / BPU (GHz) | CPU threads |
| --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Large-100 | S100 | 1.065 | 923 / 1,243 | 302 / 383 | 1.5 / 1.0 | 6 |
| MobileNetV3-Large-100 | S100P | 0.853 | 1,148 / 1,601 | 389 / 496 | 2.0 / 1.5 | 6 |
| MobileNetV3-Large-100 | S600 | 0.709 | 1,383 / 2,688 | 567 / 1,016 | 2.1 / 1.5 | 18 |
| MobileNetV3-Small-100 | S100 | 0.670 | 1,458 / 2,428 | 320 / 379 | 1.5 / 1.0 | 6 |
| MobileNetV3-Small-100 | S100P | 0.560 | 1,731 / 2,989 | 433 / 529 | 2.0 / 1.5 | 6 |
| MobileNetV3-Small-100 | S600 | 0.497 | 1,959 / 3,806 | 650 / 1,071 | 2.1 / 1.5 | 18 |

The CPU governor was `performance` and the CPUs ran at the clock listed (all
online cores) during each measurement; the boards differ in CPU and BPU clocks,
so compare targets with care. The C++ pipeline uses a scalar, bit-exact
reimplementation of Pillow's bicubic resize, which dominates its preprocessing
time; the latency columns record this implementation's measured cost. To reproduce the accuracy numbers see the
[evaluator](evaluator/README.md); the C++ timing tool is described in the
[benchmark instructions](../../../utils/tools/classification/cpp/README.md).

![Inference result](./test_data/inference.png)

*Reference inference result of the MobileNetV3-large model on an RDK S100, written with
`--img-save-path`: the bundled
[kit_fox.JPEG](test_data/kit_fox.JPEG) ranks `kit fox` first (score 0.855), followed by red fox, dingo, grey fox, and lion.*

<a id="entry-points"></a>
## Entry points

- Model preparation: [model/README.md](model/README.md)
- Python runtime: [runtime/python/README.md](runtime/python/README.md)
- Conversion: [conversion/README.md](conversion/README.md)
- Evaluation: [evaluator/README.md](evaluator/README.md)

<a id="license"></a>
## License

Sample code follows the repository top-level LICENSE (Apache-2.0). The
source model is the upstream MobileNetV3 distribution; upstream model/weights
licensing is governed by that distribution (see the reference-implementation
link above). Published artifacts follow the platform release manifests; the
manifests carry no separate license field, and no additional license is
claimed here.
