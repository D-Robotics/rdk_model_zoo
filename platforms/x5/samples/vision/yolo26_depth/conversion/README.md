[English](./README.md) | [简体中文](./README_cn.md)

# YOLO26 Depth Model Conversion and Compilation Guide

This directory provides the scripts, resources, and instructions for exporting Ultralytics YOLO26 Depth models, running quantized compilation, and checking `.bin` artifacts for RDK X5 BPU deployment.

## Directory Structure

```text
.
├── export.py                  # Ultralytics YOLO26 Depth ONNX export script
├── extract_sunrgbd_subset.py  # Deterministic SUN RGB-D subset extraction utility
├── prepare_calibration.py     # Calibration tensor preparation utility
├── mapper.py                  # Prepare Mapper YAML and invoke X5 OpenExplorer compiler
├── ptq_yamls/                 # Reference PTQ YAML files for N/S/M/L/X variants
├── requirements.txt           # Python dependencies for export and calibration utilities
├── README.md
└── README_cn.md
```

## Compilation Environment

Run model compilation on an x86 Linux host with the corresponding RDK X5 OpenExplorer environment. Installing or running the compiler toolchain on the RDK X5 board is not recommended.

Toolchain entry points:

- RDK X5 OpenExplorer / algorithm toolchain documentation: <https://developer.d-robotics.cc/rdk_doc/Advanced_development/toolchain_development/overview>
- OE toolchain download and manuals: <https://toolchain.d-robotics.cc/>

### 1. Install Docker

Install Docker by following the official Docker documentation and verify the installation:

```bash
sudo docker --version
sudo docker run --rm hello-world
```

### 2. Download and Load the X5 Toolchain Offline Image

Use the RDK X5 OpenExplorer offline image that matches the expected compiler version. The validation data for this sample was generated with OpenExplorer v1.2.8 / Mapper 1.24.3.

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker images
```

### 3. Start the Container

Mount the repository and an external work directory into the container. Increase shared memory to reduce compilation failures on large variants.

```bash
# Assume the current directory is the rdk_model_zoo_mc_rdkx5 repository root
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  -v /path/to/external/work:/work \
  --workdir /workspace \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

Use the actual image name and tag shown by `sudo docker images` when they differ.

### 4. Verify Toolchain and Python Dependencies

```bash
hb_mapper --version
hb_model_info --help
python3 -m pip install -r samples/vision/yolo26_depth/conversion/requirements.txt
```

## Conversion Flow

### High-Performance Computing Process Introduction

#### Monocular Depth Estimation

YOLO26 Depth predicts a dense relative-depth map from one RGB image. The original floating-point model contains task-specific decode and refinement logic that is convenient for training but not ideal for direct board deployment.

For deployment, the conversion keeps the BPU graph focused on the heavy convolutional backbone, neck, and depth head. The exported graph returns a low-resolution calibrated log-depth tensor with shape `1x192x192x1` for a `768x768` input. Runtime code then performs the lightweight postprocess on CPU:

- exponentiate the calibrated log-depth tensor with `exp(log_depth)`;
- resize the `192x192` depth map back to the padded model input size;
- remove 114-value letterbox padding and restore the original image geometry;
- colorize or serialize the restored relative-depth output.

This split keeps the compiled model simple, preserves the runtime tensor protocol, and avoids committing generated intermediate artifacts. The Mapper configuration also keeps the tail depth convolution output as `int16` to reduce quantization loss in the log-depth map.

### 1. Environment Preparation and Model Training

This operation is performed on an x86 machine. Ubuntu 22.04 with Python 3.10 is recommended. A GPU-enabled environment is useful for training or validating custom weights, but compilation itself is handled by the X5 toolchain.

Download the `ultralytics/ultralytics` repository and configure the training/export environment by following the official Ultralytics documentation.

```bash
git clone https://github.com/ultralytics/ultralytics.git
```

For model training, follow the official Ultralytics Depth documentation. The source `.pt` weight should be trained with the `ultralytics/ultralytics` repository, or you can use compatible pretrained YOLO26 Depth weights. No program changes are required during training, and the model `forward` method should not be modified in the training repository.

Ultralytics documentation:

- Quick Start: <https://docs.ultralytics.com/quickstart/>
- Model Training: <https://docs.ultralytics.com/modes/train/>
- Depth Task: <https://docs.ultralytics.com/tasks/depth/>

### 2. Export ONNX

This operation is performed on an x86 machine in the Ultralytics training/export environment. Prepare a YOLO26 Depth `.pt` file and run `export.py` from this directory.

`export.py` uses `ultralytics.YOLO` to load the `.pt` model, applies Python-side patches for X5-friendly export, and calls `ultralytics.YOLO.export`. The exported ONNX and `export-report.json` are written to the external output directory.

```bash
cd samples/vision/yolo26_depth/conversion

python3 export.py \
  --weights /work/weights/yolo26n-depth.pt \
  --variant n \
  --imgsz 768 \
  --opset 11 \
  --output-dir /work/yolo26_depth/export_n
```

The export script supports variants `n`, `s`, `m`, `l`, and `x`. Use the matching weight file and `--variant` value for each model size.

### 3. Prepare Calibration Data

The validated configuration uses 100 deterministic SUN RGB-D training images. Calibration tensors are RGB CHW uint8 with the same 114-value letterbox policy as runtime preprocessing. Mapper applies `data_scale=1/255` during compilation.

Extract a deterministic SUN RGB-D subset:

```bash
python3 extract_sunrgbd_subset.py \
  --archive /work/datasets/SUNRGBD.zip \
  --split train \
  --count 100 \
  --seed 20260725 \
  --output /work/yolo26_depth/sunrgbd_train100
```

Pack calibration binaries:

```bash
python3 prepare_calibration.py \
  --images /work/yolo26_depth/sunrgbd_train100/images \
  --output /work/yolo26_depth/calibration_768 \
  --count 100 \
  --seed 20260725 \
  --size 768 \
  --manifest /work/yolo26_depth/calibration_768_manifest.json \
  --report /work/yolo26_depth/calibration_768_report.md
```

Equivalent representative RGB images can be used when SUN RGB-D is unavailable, but keep the same preprocessing and calibration count policy when comparing results.

### 4. Model Compilation

Run model compilation in the RDK X5 OpenExplorer toolchain environment. Prepare the exported ONNX model and packed calibration directory before running `mapper.py`.

`mapper.py` generates the YAML configuration, runs `hb_mapper checker`, runs `hb_mapper makertbin`, copies the final `.bin` and quantized ONNX to `artifacts/`, and writes logs plus `compile-report.json` under `reports/`.

```bash
cd samples/vision/yolo26_depth/conversion

python3 mapper.py \
  --onnx /work/yolo26_depth/export_n/yolo26n-depth-log.onnx \
  --variant n \
  --calibration /work/yolo26_depth/calibration_768 \
  --size 768 \
  --optimize-level O3 \
  --output /work/yolo26_depth/compile_n
```

The script exposes common parameters, and the defaults cover the validated configuration.

```bash
$ python3 mapper.py -h
usage: mapper.py [-h] --onnx ONNX --variant {n,s,m,l,x} --calibration CALIBRATION --output OUTPUT --size {768} [--jobs JOBS] [--optimize-level {O0,O1,O2,O3}]

options:
  -h, --help                        show this help message and exit
  --onnx ONNX                       exported floating-point ONNX model path
  --variant {n,s,m,l,x}             YOLO26 Depth model size
  --calibration CALIBRATION         packed calibration binary directory
  --output OUTPUT                   new external output directory
  --size {768}                      model input size; this sample validates 768 only
  --jobs JOBS                       Mapper compilation jobs, default: 16
  --optimize-level {O0,O1,O2,O3}    compiler optimization level, default: O3
```

Recommended `.bin` file names:

- `yolo26n_depth_bayese_768x768_nv12.bin`
- `yolo26s_depth_bayese_768x768_nv12.bin`
- `yolo26m_depth_bayese_768x768_nv12.bin`
- `yolo26l_depth_bayese_768x768_nv12.bin`
- `yolo26x_depth_bayese_768x768_nv12.bin`

Place model files under `model/bayes-e/` in this sample so that `runtime/python/run.sh`, `runtime/cpp/run.sh`, and their `main` programs can use them directly.

## Input and Output Protocol

### Input Protocol

The runtime uses one NV12 pyramid input tensor named `images`.

- Training/export layout before Mapper preprocessing: `NCHW` RGB.
- Runtime layout: `NHWC` NV12 pyramid.
- Validated input size: `768x768`.
- Calibration preprocessing: 114-value letterbox padding and `data_scale=1/255`.

Converted models must keep this input protocol, otherwise the Python and C++ runtime samples will fail shape or tensor-type checks.

### Output Protocol

The runtime expects one dequantized float32 output tensor:

- output shape: `1x192x192x1`;
- semantic meaning: calibrated log-depth;
- postprocess: `depth = exp(log_depth)`, bilinear resize, letterbox restoration;
- final output: dense relative depth at the source image resolution.

The output is relative depth rather than calibrated metric depth. Dataset-level accuracy evaluation therefore requires scale or scale-shift alignment before comparing with metric ground truth.

## Check Compilation Results

Use `hb_model_info` or `hrt_model_exec` to inspect the generated `.bin` model. Performance numbers in `../evaluator/README.md` were measured on the RDK X5 board and cover model execution only.

```bash
hb_model_info /work/yolo26_depth/compile_n/artifacts/yolo26n_depth_bayese_768x768_nv12.bin
hrt_model_exec model_info --model_file /work/yolo26_depth/compile_n/artifacts/yolo26n_depth_bayese_768x768_nv12.bin
hrt_model_exec perf --model_file /work/yolo26_depth/compile_n/artifacts/yolo26n_depth_bayese_768x768_nv12.bin --thread_num 1
```

A successful Mapper run writes the following files outside the repository:

```text
compile_n/
├── artifacts/
│   ├── yolo26n_depth_bayese_768x768_nv12.bin
│   └── yolo26n_depth_bayese_768x768_nv12_quantized_model.onnx
├── config/
│   └── yolo26n_depth_bayese_768x768_nv12.yaml
├── reports/
│   ├── checker.log
│   ├── makertbin.log
│   ├── hb_model_info.log
│   └── compile-report.json
└── working/
```

`compile-report.json` records model hashes, output cosine similarity, compiler latency/FPS estimates, DDR estimates, and generated artifact paths.

## FAQ

- **Permission errors**: If files copied back from Docker have unexpected ownership on the host, check file owners or run `sudo chown -R` on the external work directory.
- **Memory or IPC errors**: Add `--shm-size=15g` when starting the Docker container.
- **Unsupported optimization level**: Use `O0`, `O1`, or `O2` if the local X5 compiler package does not support `O3` for this graph.
- **Missing output cosine**: Inspect `reports/makertbin.log`; do not publish a model when Mapper did not report the output cosine value.
- **Runtime geometry mismatch**: Confirm that calibration and runtime both use the same 114-padding letterbox policy.

## License

Tools in this directory follow the repository top-level license. Ultralytics models and SUN RGB-D data remain subject to their respective upstream licenses.
