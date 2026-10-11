English | [简体中文](README_cn.md)

# MobileNetV4 image classification

MobileNetV4 is a family of image classifiers built around universal inverted bottlenecks.

Sources: [timm/models/MobileNetV4.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/MobileNetV4.py) · [MobileNetV4 -- Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518)

[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

The sample ships one Python runtime for all targets. The `MobileNetV4Classifier` class runs a `preprocess → infer → postprocess` flow chained by `predict`: it resolves one exact artifact reference from the platform release manifest for the detected board, verifies the board identity, loads `hbm_runtime` lazily, and returns a typed Top-K result
([runtime/python/README.md](runtime/python/README.md)).

### Algorithm background

MobileNetV4 unifies the block design space of mobile CNNs in the
Universal Inverted Bottleneck (UIB): depending on which depthwise layers
are enabled, one block expresses an inverted bottleneck, a
ConvNeXt-style block, an FFN-style block, or the ExtraDW variant; a
mobile multi-query attention design adds attention where it pays off on
mobile accelerators ([paper](https://arxiv.org/abs/2404.10518),
[timm/models/MobileNetV4.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/MobileNetV4.py)).

Feature summary:

- **Universal Inverted Bottleneck**: unifies inverted bottleneck, ConvNeXt-style blocks, FFN-style blocks, and ExtraDW variants.
- **Mobile Multi-Query Attention**: an attention structure optimized for mobile accelerators.
- **Model variants**: this sample ships the Conv-Small, Conv-Medium and Conv-Large deployment models.

![MobileNetV4 UIB blocks](./test_data/MobileNetV4_architecture.png)

*Universal Inverted Bottleneck blocks (Fig. 4 of the paper): the UIB
block with two optional
depthwise layers, its Extra-DW / Inverted Bottleneck / ConvNeXt / FFN
instantiations, and the alternative fused IB.*

<a id="directory"></a>
## Directory structure

```text
mobilenetv4/
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

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | small | python | supported |
| x5 | medium | python | supported |
| x5 | large | python | supported |
| s100 | small | python | supported |
| s100 | medium | python | supported |
| s100 | large | python | supported |
| s100p | small | python | supported |
| s100p | medium | python | supported |
| s100p | large | python | supported |
| s600 | small | python | supported |
| s600 | medium | python | supported |
| s600 | large | python | supported |

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
python3 -m venv .venv-mobilenetv4
source .venv-mobilenetv4/bin/activate
python3 -m pip install -r samples/vision/mobilenetv4/requirements-host.txt
python3 -c "import cv2, numpy, yaml, PIL; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:mobilenetv4:mobilenetv4_conv_small_bayese_224x224_nv12.bin)
#    output: samples/vision/mobilenetv4/model/mobilenetv4_conv_small_bayese_224x224_nv12.bin
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/mobilenetv4/model/download.sh x5 small

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/mobilenetv4/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv4:mobilenetv4_conv_small_bayese_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv4/model/mobilenetv4_conv_small_bayese_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv4/test_data/great_grey_owl.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

For S100, S100P and S600 use the matching `s:` reference (see `--list-models`) and the
same root `datasets/imagenet/` labels. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; Pass `--img-save-path` to save a visualization; otherwise results are printed to stdout. With the bundled `great_grey_owl.JPEG` the Top-1 is class 24 (`great grey owl`)
and with `zebra_cls.jpg` it is class 340 (`zebra`), for all three variants on X5, S100, S100P
and S600 (checked on each board with the published builds). Select a target and variant listed in the [Support matrix](#support-matrix), prepare that exact manifest artifact with the model downloader, and run the sample on the matching board.

<a id="performance"></a>
## Performance data

All numbers were measured on real boards with the published artifacts
(INT8, 224x224 or 256x256, batch 1). The models are 3.77 M parameters / 0.37 GFLOPs (Small), 9.72 M parameters / 1.66 GFLOPs (Medium) and 32.59 M parameters / 5.67 GFLOPs (Large);
GFLOPs counts Conv and Gemm multiply-accumulates as two operations.

**Accuracy.** Top-1 / Top-5 over the complete ImageNetV2 MatchedFrequency set
(10,000 images, 1,000 classes). This is not the ILSVRC2012 validation set, so
the values are not comparable with ImageNet-1k validation figures. "FP32" is
the ONNX export of the same checkpoint evaluated on the same crops; "Board" is
the compiled model on the matching board.

| Model | Target | FP32 Top-1 | Board Top-1 | FP32 Top-5 | Board Top-5 |
| --- | --- | --- | --- | --- | --- |
| MobileNetV4-Conv-Small | X5 | 60.96% | 58.72% | 82.64% | 80.53% |
| MobileNetV4-Conv-Small | S100 | 60.96% | 58.56% | 82.64% | 80.78% |
| MobileNetV4-Conv-Small | S100P | 60.96% | 58.56% | 82.64% | 80.78% |
| MobileNetV4-Conv-Small | S600 | 60.96% | 58.57% | 82.64% | 80.94% |
| MobileNetV4-Conv-Medium | X5 | 67.35% | 66.25% | 87.86% | 87.54% |
| MobileNetV4-Conv-Medium | S100 | 67.35% | 66.94% | 87.86% | 87.65% |
| MobileNetV4-Conv-Medium | S100P | 67.35% | 66.94% | 87.86% | 87.65% |
| MobileNetV4-Conv-Medium | S600 | 67.35% | 66.75% | 87.86% | 87.56% |
| MobileNetV4-Conv-Large | X5 | 70.79% | 69.85% | 89.15% | 89.13% |
| MobileNetV4-Conv-Large | S100 | 70.79% | 69.66% | 89.15% | 89.04% |
| MobileNetV4-Conv-Large | S100P | 70.79% | 69.66% | 89.15% | 89.04% |
| MobileNetV4-Conv-Large | S600 | 70.79% | 69.62% | 89.15% | 89.16% |

**Speed.** Runtime numbers come from `hrt_model_exec perf` on BPU core 0 (the
model only: no preprocessing or postprocessing); FPS is the total completed
frames divided by the common wall time of 3 runs x 200 frames after a 20-frame
warmup. The C++ pipeline column times one frame from an in-memory BGR image to
a Top-5 list, including resize, crop, NV12 packing, input upload, inference,
and ranking (file reading and decoding excluded), for 1 and 2 independent
streams.

| Model | Target | Runtime latency, 1 thread (ms) | Runtime FPS, 1 / 2 threads | C++ pipeline FPS, 1 / 2 streams | CPU / BPU (GHz) | CPU threads |
| --- | --- | --- | --- | --- | --- | --- |
| MobileNetV4-Conv-Small | X5 | 0.999 | 994 / 1,428 | 108 / 119 | 1.5 / 1.0 | 8 |
| MobileNetV4-Conv-Small | S100 | 0.407 | 2,359 / 4,203 | 362 / 416 | 1.5 / 1.0 | 6 |
| MobileNetV4-Conv-Small | S100P | 0.355 | 2,710 / 4,501 | 459 / 534 | 2.0 / 1.5 | 6 |
| MobileNetV4-Conv-Small | S600 | 0.311 | 3,078 / 5,991 | 741 / 911 | 2.1 / 1.5 | 18 |
| MobileNetV4-Conv-Medium | X5 | 2.066 | 483 / 564 | 101 / 115 | 1.5 / 1.0 | 8 |
| MobileNetV4-Conv-Medium | S100 | 0.607 | 1,595 / 2,831 | 366 / 431 | 1.5 / 1.0 | 6 |
| MobileNetV4-Conv-Medium | S100P | 0.530 | 1,829 / 3,174 | 461 / 551 | 2.0 / 1.5 | 6 |
| MobileNetV4-Conv-Medium | S600 | 0.407 | 2,381 / 4,605 | 707 / 980 | 2.1 / 1.5 | 18 |
| MobileNetV4-Conv-Large | X5 | 5.423 | 184 / 195 | 71 / 87 | 1.5 / 1.0 | 8 |
| MobileNetV4-Conv-Large | S100 | 1.147 | 858 / 1,120 | 285 / 359 | 1.5 / 1.0 | 6 |
| MobileNetV4-Conv-Large | S100P | 1.063 | 926 / 1,188 | 346 / 458 | 2.0 / 1.5 | 6 |
| MobileNetV4-Conv-Large | S600 | 0.631 | 1,553 / 2,829 | 582 / 1,017 | 2.1 / 1.5 | 18 |

The CPU governor was `performance` and the CPUs ran at the clock listed (all
online cores) during each measurement; the boards differ in CPU and BPU clocks,
so compare targets with care. The C++ pipeline uses a scalar, bit-exact
reimplementation of Pillow's bicubic resize, which dominates its preprocessing
time; it is not an upper bound for an optimized pipeline. To reproduce the accuracy numbers see the
[evaluator](evaluator/README.md); the C++ timing tool is described in the
[benchmark instructions](../../../utils/tools/mobilenet/cpp/README.md).

![Inference result](./test_data/inference.png)

*Reference inference result of the Small model on an RDK X5, written with
`--img-save-path`: the bundled
[great_grey_owl.JPEG](test_data/great_grey_owl.JPEG) ranks `great grey owl`
first (score 0.957), followed by vulture, cheetah, lynx, and prairie chicken.*

<a id="entry-points"></a>
## Entry points

- Model preparation: [model/README.md](model/README.md)
- Python runtime: [runtime/python/README.md](runtime/python/README.md)
- Conversion: [conversion/README.md](conversion/README.md)
- Evaluation: [evaluator/README.md](evaluator/README.md)

<a id="license"></a>
## License

Sample code follows the repository top-level LICENSE (Apache-2.0). The
source model is the upstream MobileNetV4 distribution; upstream model/weights
licensing is governed by that distribution (see the reference-implementation
link above). Published artifacts follow the platform release manifests; the
manifests carry no separate license field, and no additional license is
claimed here.
