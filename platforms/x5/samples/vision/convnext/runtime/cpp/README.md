English | [简体中文](./README_cn.md)

# ConvNeXt C++ Runtime

This directory keeps the C++ runtime placeholder for the ConvNeXt sample.

## Overview

The maintained runtime path of this sample is `runtime/python`.

The current `runtime/cpp` directory does not provide a complete C++ inference implementation. It only keeps placeholder files so the sample structure remains aligned with the repository template.

## Directory Structure

```text
.
├── CMakeLists.txt
└── run.sh
```

## Build

The current `CMakeLists.txt` only prints a placeholder message and does not build a runnable executable.

```bash
mkdir -p build
cd build
cmake ..
```

## Run

The current `run.sh` reports that the C++ runtime is not implemented for this sample.

```bash
chmod +x run.sh
./run.sh
```

## Notes

- Use `runtime/python/main.py` for actual inference.
- Do not treat this directory as a maintained C++ runtime path.
