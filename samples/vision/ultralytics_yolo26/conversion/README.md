# YOLO26 Model Conversion Guide

English | [简体中文](./README_cn.md)

This directory provides tools and instructions for converting YOLO26 models (from the Ultralytics framework) to quantized BPU `.bin` models compatible with D-Robotics RDK hardware.

## Model Compilation Environment

To convert models, you need the **RDK X5 OpenExplore Toolchain**. We provide two installation methods.

### Method 1: Pip Installation (Recommended)

This method installs a trimmed version of the toolchain directly on your x86 Linux machine. It is lightweight and easy to set up using Conda.

**Note**: Perform this on an x86 machine (Ubuntu 22.04 recommended). **Do not** install this on the RDK board itself.

1.  **Set up a Python Environment (Miniconda)**
    It is highly recommended to use a virtual environment to avoid conflicts.
    ```bash
    # Create a new environment named 'rdk_env' with Python 3.10
    conda create -n rdk_env python=3.10 -y
    
    # Activate the environment
    conda activate rdk_env
    ```

2.  **Install the Toolchain**
    ```bash
    pip install rdkx5-yolo-mapper
    ```
    *(Optional) Use a mirror if download is slow:*
    ```bash
    pip install rdkx5-yolo-mapper -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
    ```

3.  **Verify Installation**
    ```bash
    hb_mapper --version
    # Expected output: hb_mapper, version 1.24.3 (or newer)
    ```

**Troubleshooting Download Issues**:
If you see errors like `incomplete-download` or `Download failed because not enough bytes were received`, it is likely due to network instability. Simply retry the installation command. Pip will skip already installed packages.

---

### Method 2: Docker Installation (Alternative)

If you prefer an isolated container environment or encounter dependency issues with Method 1, use the official Docker image.

**RDK X5 OpenExplore 1.2.8**
```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```
Or download the offline image from the [D-Robotics Developer Community](https://forum.d-robotics.cc/t/topic/28035).

**Running the Container**:
```bash
# Mount your model zoo directory into the container
docker run -it --rm -v /path/to/rdk_model_zoo:/data openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

---

## Conversion Workflow

### 1. One-Key Conversion (Recommended)

We provide a `mapper.py` script that automates the entire process: preparing calibration data, generating configuration files, and running `hb_mapper`.

**Prerequisites**:
- An ONNX model exported for BPU (see `onnx_export/`).
- A folder with 20~50 images (`.jpg`, `.png`) for calibration.

**Command**:
```bash
python3 mapper.py --onnx [model.onnx] --cal-images [image_folder_path]
```
The converted `.bin` model will be generated in the same directory as the ONNX model.

### 2. Mapper Arguments

The `mapper.py` script exposes common parameters for customization:

```bash
python3 mapper.py -h
```

| Argument | Description |
| :--- | :--- |
| `--onnx` | Path to the source float ONNX model. |
| `--cal-images` | Directory containing calibration images (20~50 recommended). |
| `--quantized` | Precision level: `int8` (default, recommended) or `int16`. |
| `--jobs` | Number of parallel compilation jobs. |
| `--optimize-level` | Compiler optimization level: `O0`, `O1`, `O2` (default), `O3`. |
| `--cal-sample` | Whether to sample images from the folder (default: True). |
| `--save-cache` | Whether to keep temporary BPU output files (default: False). |

---

## License
Tools in this directory follow the [Apache 2.0 License](../../../../LICENSE).