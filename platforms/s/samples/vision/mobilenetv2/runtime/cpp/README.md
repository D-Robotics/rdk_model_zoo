English | [简体中文](./README_cn.md)

# MobileNetV2 Image Classification Sample (C++)

This sample demonstrates how to run a quantised MobileNetV2 model on the BPU for image classification, outputting Top-K class labels with confidence scores.

## Dependencies

Make sure the following packages are installed:

```bash
sudo apt update
sudo apt install libgflags-dev
```

## Directory Structure

```bash
.
|-- inc                    # Header directory
|   `-- mobilenetv2.hpp    # MobileNetV2 wrapper class declaration
|-- src                    # Source directory
|   |-- main.cpp           # Inference entry (argument parsing & flow control)
|   `-- mobilenetv2.cpp    # MobileNetV2 inference & post-processing
|-- CMakeLists.txt         # CMake build configuration
|-- README.md              # C++ inference guide
`-- run.sh                 # Run script
```

## Build

- Configure and compile:

    ```bash
    mkdir build && cd build
    cmake ..
    make -j$(nproc)
    ```

## Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--model_path` | Path to `.hbm` model file | `/opt/hobot/model/<soc>/basic/mobilenetv2_224x224_nv12.hbm` |
| `--test_img` | Test image path | `../../../test_data/zebra_cls.jpg` |
| `--label_file` | Label file path | `../../../test_data/imagenet1000_labels.txt` |
| `--top_k` | Number of Top-K results to print | `5` |

> **Note**: The `<soc>` in the default `--model_path` is injected by CMake at build time based on the current board SoC (e.g. `s100`, `s600`).

## Quick Run

- Run the model
    - One-click via script (auto-installs dependencies, downloads model, builds, and runs):

        ```bash
        ./run.sh
        ```
    - Run with defaults (inside `build/`):

        ```bash
        ./mobilenetv2
        ```
    - Run with custom arguments:

        ```bash
        ./mobilenetv2 \
            --model_path /opt/hobot/model/s100/basic/mobilenetv2_224x224_nv12.hbm \
            --test_img ../../../test_data/zebra_cls.jpg \
            --label_file ../../../test_data/imagenet1000_labels.txt \
            --top_k 5
        ```

- View results

    On success, the Top-K classification results are printed to the terminal:

    ```bash
    TOP-1: label=zebra, prob=0.992246
    TOP-2: label=tiger, Panthera tigris, prob=0.00404656
    TOP-3: label=hartebeest, prob=0.00133707
    TOP-4: label=tiger cat, prob=0.000722661
    TOP-5: label=impala, Aepyceros melampus, prob=0.000539704
    ```

## API Reference

See the [source reference docs](../../../../../docs/source_reference/README.md) for detailed API documentation.