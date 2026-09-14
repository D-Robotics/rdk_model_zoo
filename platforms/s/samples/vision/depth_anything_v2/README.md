English | [简体中文](./README_cn.md)

# Depth Anything V2 Model Description

Depth Anything V2 is a monocular depth estimation model. This sample provides HBM model download, Python inference, conversion notes, and evaluation records for RDK S100/S100P. The sample input is `furseal.jpg`, and the output is a colorized depth map.

## Algorithm Overview

Depth Anything is a practical monocular depth estimation solution that aims to build a foundation model capable of handling arbitrary images without introducing complex new modules. Compared with V1, Depth Anything V2 produces more refined and robust depth predictions through three key practices: replacing all labeled real images with synthetic images, increasing teacher model capacity, and training the student model with large-scale pseudo-labeled real images.

![Depth Anything V2 framework](./test_data/readme_img/image-2.png)

- Project website: <https://depth-anything.github.io/>
- Paper: <https://arxiv.org/abs/2406.19675>
- Official repository: <https://github.com/DepthAnything/Depth-Anything-V2>

## Algorithm Capabilities

- Monocular depth estimation
- Dense depth map output
- Colorized depth visualization saving

## Algorithm Features

- The input is an NCHW RGB image tensor.
- The output is a single-channel depth map.
- Postprocessing uses bilinear interpolation to recover the original image size and normalizes values to `[0, 255]`.

## Directory Structure

```text
depth_anything_v2/
├── conversion/
├── evaluator/
├── model/
├── runtime/
│   └── python/
├── test_data/
│   ├── furseal.jpg
│   └── readme_img/
├── README.md
└── README_cn.md
```

## Quick Start

```bash
cd samples/vision/depth_anything_v2/runtime/python
bash run.sh
```

The script downloads `../../model/s100/depth_any.hbm`, reads `../../test_data/furseal.jpg`, and saves `result.jpg`.

## Model Conversion

See [conversion/README.md](./conversion/README.md) for ONNX input/output, operator notes, int16 quantization accuracy, and OE toolchain entry points.

## Runtime

See [runtime/python/README.md](./runtime/python/README.md) for Python runtime arguments and direct `python3 main.py` examples.

## Model Evaluation

See [evaluator/README.md](./evaluator/README.md) for performance data, reference depth results, and board monitoring metrics.

## Inference Result

The HBM model output depth map should match the spatial structure of the input image and show clear depth differences between foreground and background.

![Depth Anything V2 depth result](./test_data/readme_img/depth_color.png)

## License

This sample is licensed under the [Apache 2.0 License](../../../LICENSE).
