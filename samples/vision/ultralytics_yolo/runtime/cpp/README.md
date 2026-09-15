English | [简体中文](./README_cn.md)

# Ultralytics YOLO C++ Sample

This directory keeps the C++ reference runtime for Ultralytics YOLO on RDK X5.

## Overview

The recommended runtime path of this sample is `runtime/python`. The C++ code
in this directory is kept as a reference implementation.

## Directory Structure

```bash
.
|-- classify/   # Classification reference
|-- detect/     # Detection reference
|-- pose/       # Pose reference
`-- segment/    # Segmentation reference
```

Each subdirectory contains its own `main.cc` and `CMakeLists.txt`.

## Build

Use the task subdirectory that you want to inspect or build.

Detection example:

```bash
cd runtime/cpp/detect
mkdir -p build
cd build
cmake ..
make
```

Classification example:

```bash
cd runtime/cpp/classify
mkdir -p build
cd build
cmake ..
make
```

## Run

Run the generated executable in the corresponding `build/` directory after the
build is finished.

## Notes

- This directory is preserved for reference.
- The recommended runtime entry is `runtime/python/main.py`.
