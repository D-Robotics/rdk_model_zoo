English | [简体中文](README_cn.md)

# Faster R-CNN

<a id="overview"></a>
## Overview

[TorchVision Faster R-CNN](https://github.com/pytorch/vision/tree/v0.21.0/torchvision/models/detection) uses a region proposal network and a region classification head for COCO object detection. This sample runs the MobileNetV3-Large-320-FPN model and adds ByteTrack person tracking for video. The backbone/RPN and box head run on BPU; CPU handles proposal selection, RoIAlign and tracking.

<a id="directory"></a>
## Directory structure

```text
fasterrcnn/
├── conversion/       # Weight export, calibration and board compilation
├── model/            # Published-pair download and install, target metadata, COCO labels
├── runtime/python/   # main.py, cli.py, detect.py, tracking.py, run.sh
├── runtime/cpp/      # Three source files, two headers, CMakeLists.txt, run.sh
└── test_data/        # bus.jpg image input
```

<a id="support-matrix"></a>
## Supported boards

| Target | Model | Python | C++ | Runtime SDK |
| --- | --- | --- | --- | --- |
| `x5` | MobileNetV3-Large-320-FPN, PartA + PartB `.bin` | ✓ | ✓ | DNN 1.24.5 |
| `s100` | MobileNetV3-Large-320-FPN, PartA + PartB `.hbm` | ✓ | ✓ | DNN/UCP 3.13.6, HBRT 4.7.5 |
| `s100p` | MobileNetV3-Large-320-FPN, PartA + PartB `.hbm` | ✓ | ✓ | DNN/UCP 3.13.6, HBRT 4.7.5 |
| `s600` | MobileNetV3-Large-320-FPN, PartA + PartB `.hbm` | ✓ | ✓ | DNN/UCP 3.13.6, HBRT 4.7.5 |

<a id="prerequisites"></a>
## Prerequisites

Use the board image’s `hbm_runtime`, NumPy, OpenCV and SciPy. X5/S100/S100P use Python 3.10; S600 uses Python 3.12. For person tracking, create a virtual environment from the repository root and install the two tracker packages. Source builds also need a C++ compiler and the installed NumPy headers.
```bash
python3 -m venv --system-site-packages .venv-fasterrcnn
. .venv-fasterrcnn/bin/activate
python3 -m pip install Cython setuptools wheel PyYAML==6.0.3
python3 -m pip install --no-build-isolation --no-deps lap==0.5.12 cython-bbox==0.1.5
```

Model conversion is only needed when self-compiling; it uses an x86_64 Linux computer with Docker and the OE images listed in the [conversion guide](conversion/README.md#toolchain-targets). Keep both model stages from the same target together.

<a id="quickstart"></a>
## Quick start

Download the published pair, copy the repository to the board and run the Python runtime. On a computer with internet access, from the repository root:
The download computer needs Python 3.10 or newer.
```bash
python3 -m venv .venv-model-download
. .venv-model-download/bin/activate
python3 -m pip install PyYAML==6.0.3
python3 samples/vision/fasterrcnn/model/download.py --target x5
```
The command downloads `frcnn_mbv3_320_parta.bin` and `frcnn_mbv3_320_partb.bin` from the public archive into `model/x5/` and verifies both SHA-256 digests against `model/frcnn_mbv3_320.json`. `--target` accepts one or more of `x5`, `s100`, `s100p`, `s600`; omitting it downloads all four pairs. Existing files are verified, never overwritten. `model/download.sh` accepts the same options.

Copy this repository, including `model/x5/`, to X5. From the repository root on the board, run:
```bash
python3 samples/vision/fasterrcnn/runtime/python/main.py --target x5
```
The command prints a JSON detection result and saves `samples/vision/fasterrcnn/test_data/result.jpg`. On the other boards use the matching target — `--target s100`, `--target s100p` or `--target s600` — after downloading that target's pair.

### Compile the pair yourself (optional)

The published pairs can also be rebuilt from the TorchVision source. Start from the repository root on the conversion computer. Obtain the OE images from the [X5 toolchain guide](https://developer.d-robotics.cc/rdk_x_doc/en/Advanced_development/toolchain_development/intermediate/environment_config) and the [S toolchain guide](https://developer.d-robotics.cc/rdk_s_doc/en/Advanced_development/toolchain_development/algorithm_toolchain/overview). Export the weights and prepare calibration inputs:
```bash
docker run --rm -v "$PWD":/work \
  -w /work/samples/vision/fasterrcnn/conversion \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0 bash -lc '
    python3 -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu &&
    python3 -m pip install numpy==1.23.0 onnx==1.15.0 onnxruntime==1.19.0 Pillow==9.3.0 &&
    python3 export_split.py &&
    python3 prepare_calibration.py --images ../test_data/bus.jpg ../../yolov5/test_data/kite.jpg --coco-urls recorded_calibration_urls.txt --hflip repo
  '
```
Compile the X5 pair:
```bash
docker run --rm -v "$PWD":/work \
  -w /work/samples/vision/fasterrcnn/conversion \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310 bash -lc \
  'python3 make_configs.py && bash compile_models.sh x5'
```
Copy the pair into its model directory:
```bash
python3 samples/vision/fasterrcnn/model/prepare.py --target x5 \
  --from-build samples/vision/fasterrcnn/conversion/build
```
Run the self-compiled pair from the repository root on X5:
```bash
python3 samples/vision/fasterrcnn/runtime/python/main.py --target x5 \
  --parta-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_partb.bin
```
For S100/S100P/S600, follow the OE container, compilation and installation commands in the [conversion guide](conversion/README.md#compile).

<a id="expected-results"></a>
## Results

`bus.jpg` produces 22 detections on X5/S100/S100P and 17 on S600 with the listed model pairs. Results contain original-image `xyxy` boxes, scores and COCO category IDs. Person has ID 1. For video, pass `--video` and `--records`; the runtime writes an annotated MP4 and frame-by-frame track records.

MobileNetV3-Large-320-FPN, 320×320 canvas, up to 150 proposals, 30 ordered 1280×720 frames with about 7–9 active tracks, `--threads 1 --no-encode`. Values are mean wall time. `predict` covers image preprocessing, both BPU graphs, CPU RPN/RoIAlign, detection decoding and person tracking. Tracker time is the CPU association subset of `predict` time. Video decoding and encoding are excluded from these two columns. SDK versions are in the support table.

| Board | Python predict (ms/frame) | C++ predict (ms/frame) | Python tracker (ms/frame) | C++ tracker (ms/frame) |
| --- | ---: | ---: | ---: | ---: |
| X5 8 GB | 2168.65 | 218.74 | 6.57 | 0.157 |
| S100 | 483.81 | 44.71 | 2.10 | 0.041 |
| S100P | 387.14 | 33.79 | 1.67 | 0.034 |
| S600 64 GB | 259.27 | 32.56 | 1.51 | 0.033 |

`--threads 2` and `--threads 4` set OpenCV’s CPU thread count for image conversion and resize. The same clip had similar total prediction times at all three settings.

<a id="entry-points"></a>
## Guides

[Model preparation](model/README.md) · [Python runtime](runtime/python/README.md) · [C++ runtime](runtime/cpp/README.md) · [Conversion](conversion/README.md).

<a id="license"></a>
## Licenses

Sample code follows the repository [LICENSE](../../../LICENSE). TorchVision source uses [BSD-3-Clause](https://github.com/pytorch/vision/blob/v0.21.0/LICENSE); weights come from the official COCO_V1 release. ByteTrack and LAP source attribution is in the [tracker source map](../bytetrack/runtime/python/TRACKER_SOURCE_MAP.json) and [C++ third-party licenses](../bytetrack/runtime/cpp/THIRD_PARTY_LICENSES.md). Dataset terms are listed by [COCO](https://cocodataset.org/#termsofuse).

