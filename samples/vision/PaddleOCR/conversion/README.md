[English](./README.md) | [简体中文](./README_cn.md)

# Model Conversion

This document describes how to convert pre-trained models to D-Robotics BPU executable BIN model files.

---

## Directory Structure

```
conversion/
├── README.md              # English
├── README_cn.md           # Chinese
└── ptq_yamls/             # PTQ configuration files
    ├── paddleocr_det_config.yaml  # Detection model config
    └── paddleocr_rec_config.yaml  # Recognition model config
```

---

## Conversion Workflow

```
Pre-trained Model (.pt/.onnx) → ONNX Export → Calibration Data Prep → PTQ Quantization → Compile to BIN
```

---

## Build Environment

To convert models, you need to install the **RDK X5 OpenExplore Toolchain**. We provide two installation methods, **Method 1 is recommended**.

### Method 1: Pip Installation (Recommended)

This method installs a lightweight toolchain on x86 Linux machines, recommended to use with Miniconda.

**Note**: This operation is only performed on x86 development machines (Ubuntu 22.04 recommended), **never** install on RDK board.

1. **Create Python Environment (Miniconda)**
   
   Strongly recommend using virtual environment to avoid dependency conflicts.
   ```bash
   # Create Python 3.10 environment named rdk_env
   conda create -n rdk_env python=3.10 -y
   
   # Activate environment
   conda activate rdk_env
   ```

2. **Install Toolchain**
   
   ```bash
   pip install rdkx5-yolo-mapper
   ```
   
   *(Optional) If download is slow, use Aliyun mirror:*
   ```bash
   pip install rdkx5-yolo-mapper -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
   ```

3. **Verify Installation**
   
   ```bash
   hb_mapper --version
   # Expected output: hb_mapper, version 1.24.3 (or newer)
   ```

**FAQ**: If you encounter `incomplete-download` or download failure errors, it's usually due to unstable network. Re-run the installation command, Pip will automatically skip already downloaded packages.

---

### Method 2: Docker Installation (Alternative)

If you want complete environment isolation, or encounter dependency issues with Method 1, use the official Docker image.

**RDK X5 OpenExplore 1.2.8**
```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```
Or get offline Docker image from D-Robotics Developer Community: [https://forum.d-robotics.cc/t/topic/28035](https://forum.d-robotics.cc/t/topic/28035)

**Start Container**:
```bash
# Mount your model zoo directory to container
docker run -it --rm -v /path/to/rdk_model_zoo:/data openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

---


## One-Click Conversion Script (Recommended)

We provide `mapper.py` script that automatically completes calibration data preparation, config file generation, and calls `hb_mapper` for compilation.

### Prerequisites

- BPU-adapted ONNX model already exported.
- A folder containing 20~50 images for quantization calibration (`.jpg` or `.png`).

### Run Conversion

```bash
python3 mapper.py --onnx model.onnx --cal-images /path/to/calibration/images
```

After successful conversion, the generated `.bin` model file will be in the same directory as the ONNX model.

### Script Parameters

```bash
python3 mapper.py -h
```

| Parameter | Description |
| :--- | :--- |
| `--onnx` | Path to original floating-point ONNX model. |
| `--cal-images` | Directory containing calibration images (recommended 20~50). |
| `--quantized` | Quantization precision: `int8` (default, recommended) or `int16`. |
| `--jobs` | Number of concurrent tasks during model compilation. |
| `--optimize-level` | Compiler optimization level: `O0`, `O1`, `O2` (default), `O3`. |
| `--cal-sample` | Whether to sample images from directory (default: True). |
| `--save-cache` | Whether to keep BPU compilation temporary files (default: False). |

---

## Generate Calibration Data

Model quantization requires calibration data to compute activation distributions.

### Dataset Requirements

- Quantity: Recommended 20-50 images
- Source: Typical samples from training dataset
- Format: RGB images, JPEG/PNG format
- Size: Match model input resolution (e.g., 640x640)

### Preparation Steps

```bash
# 1. Create calibration data directory
mkdir -p calibration_data_rgb_f32_640

# 2. Put images into the directory (resize to model input size)
#    Naming convention: 0001.jpg, 0002.jpg, ...

# 3. Verify data
ls calibration_data_rgb_f32_640 | wc -l
```

> **Note**: Do not use validation/test set images for calibration. Randomly select from training set.

---

## Model Check

Verify if model operators are supported by BPU before compilation:

```bash
hb_mapper checker --model-type onnx --march bayes-e --model model.onnx
```

### Common Issues

- **Unsupported operators**: Some operators will fallback to CPU, may affect performance
- **Precision issues**: Check if cosine similarity >= 0.999

---

## Model Compilation

### Compile with Configuration

```bash
hb_mapper makertbin --model-type onnx --config <config>.yaml
```

### Configuration File (ptq_yamls/*.yaml)

```yaml
model_parameters:
  onnx_model: 'model.onnx'
  model_name: 'model_name'
  march: 'bayes-e'  # RDK X5 uses bayes-e

input_parameters:
  input_name: 'images'
  input_type_rt: 'nv12'    # BPU-accelerated NV12 input
  input_space_and_range: 'regular'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'

output_parameters:
  output_name: ['output0', 'output1', 'output2']

calibration_parameters:
  cal_data_dir: './calibration_data_rgb_f32_640'
  calibration_type: 'default'
  cal_data_type: 'float32'

build_parameters:
  build_dir: './output'
  compile_mode: 'latency'  # Latency optimization
  debug: False
```

---

## Output Artifacts

After compilation, the following files are generated in the configured `build_dir`:

```
output/
├── model_name.bin           # Executable BIN model file
├── model_name_input.yaml    # Input preprocessing config
├── model_name_output.yaml   # Output postprocessing config
└── hb_mapper_makertbin.log  # Conversion log
```

### Key Log Information

The log contains important information, please keep it:
- Input/output tensor names and shapes
- Quantization parameters (scale/zero_point)
- Operator distribution (BPU/CPU)

---

## Verification

### Using hb_perf for Visualization

```bash
hb_perf model_name.bin
```

Generates performance analysis report including operator distribution, memory usage, etc.

### Using hrt_model_exec to View Model Info

```bash
hrt_model_exec model_info --model_file model_name.bin
```

Example output:
```
input[0]:
  name: images
  valid shape: (1,3,640,640,)
  tensor type: HB_DNN_IMG_TYPE_NV12

output[0]:
  name: output0
  valid shape: (1,80,80,255,)
  tensor type: HB_DNN_TENSOR_TYPE_F32
```

---

## Related Documents

- [RDK X5 Algorithm Toolchain Manual](https://developer.d-robotics.cc)
- [OpenExplorer Toolchain Download](https://developer.d-robotics.cc)
- [D-Robotics Developer Community](https://forum.d-robotics.cc)

---

## License

Tools in this directory follow [Apache 2.0 License](../../../../LICENSE).