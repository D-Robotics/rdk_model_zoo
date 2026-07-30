[English](./README.md) | [简体中文](./README_cn.md)

# C++ Runtime

## Requirements

- RDK X5 board environment with DNN Runtime headers and libraries
- CMake 3.10 or later
- C++17 compiler
- OpenCV development package

## Build And Run

Build and run with default paths:

```bash
bash run.sh
```

Specify all paths explicitly:

```bash
bash run.sh MODEL.bin INPUT.jpg OUTPUT_DIR
```

The script configures CMake under `runtime/cpp/build`, builds `yolo26_depth`, and runs one inference.

## Outputs

- `depth_native.f32`: row-major float32 relative depth at source resolution.
- `depth.png`: colorized depth visualization.
- `overlay.png`: source image and depth visualization overlay.
- `report.json`: model name, input/output sizes, and measured BPU latency.

## Code Interface

`inc/yolo26_depth.hpp` declares the reusable `Yolo26Depth` interface. `src/yolo26_depth.cpp` implements NV12 packing, DNN inference, cache synchronization, log-depth decoding, and geometry restoration.

Follow the [source-reference documentation guide](../../../../../docs/source_reference/README.md) to generate API documentation.
