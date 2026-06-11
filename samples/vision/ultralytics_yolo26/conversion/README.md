English | [简体中文](./README_cn.md)

# YOLO26 Model Conversion and Compilation Guide

This directory provides the scripts and instructions required for YOLO26 model export, quantized compilation, and HBM artifact inspection. The target artifact is a BPU-quantized `.hbm` model for RDK S100/S100P.

## Directory Structure

```text
.
├── mapper.py                       # Invokes OpenExplore compiler tools to generate HBM
├── onnx_export/                    # YOLO26 ONNX export scripts for each task
│   ├── export_yolo26_cls_bpu.py
│   ├── export_yolo26_detect_bpu.py
│   ├── export_yolo26_obb_bpu.py
│   ├── export_yolo26_pose_bpu.py
│   └── export_yolo26_seg_bpu.py
├── README.md
└── README_cn.md
```

## Model Compilation Environment

Run conversion on an x86 Linux host with the RDK S100 OpenExplore Docker environment. Do not install the compiler toolchain on the board.

Toolchain entry points:

- OE Docker documentation: <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain download: <https://toolchain.d-robotics.cc/>

### 1. Install Docker

Install Docker following the official documentation, then verify it:

```bash
sudo docker --version
sudo docker run --rm hello-world
```

### 2. Obtain and Load the Offline Image

Download the OpenExplore CPU Docker image for RDK S100/S100P from the OE Docker documentation, then load the actual image file:

```bash
sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
sudo docker images
```

### 3. Start the Container

Mount the repository and allocate enough shared memory:

```bash
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  <docker-image-name> /bin/bash
```

## Conversion Workflow

### 1. Export ONNX

Choose the export script under `onnx_export/` according to the task type and generate an ONNX model with the BPU post-processing protocol fixed:

```bash
python3 onnx_export/export_yolo26_detect_bpu.py --weights yolo26n.pt --imgsz 640
python3 onnx_export/export_yolo26_seg_bpu.py --weights yolo26n-seg.pt --imgsz 640
python3 onnx_export/export_yolo26_pose_bpu.py --weights yolo26n-pose.pt --imgsz 640
python3 onnx_export/export_yolo26_obb_bpu.py --weights yolo26n-obb.pt --imgsz 640
python3 onnx_export/export_yolo26_cls_bpu.py --weights yolo26n-cls.pt --imgsz 224
```

If script arguments change with the upstream YOLO26 version, follow `python3 <script> -h`.

### 2. Prepare Calibration Images

Prepare 20 to 50 representative `.jpg` or `.png` images:

```text
cal_images/
├── 000001.jpg
├── 000002.jpg
└── ...
```

### 3. Compile HBM Models

Run `mapper.py` from this directory. Use `--march` to select the target architecture:

```bash
cd samples/vision/ultralytics_yolo26/conversion

# RDK S100 (Nash-E)
python3 mapper.py --onnx yolo26n_detect.onnx --cal-images ./cal_images --march nash-e

# RDK S100P (Nash-M)
python3 mapper.py --onnx yolo26n_detect.onnx --cal-images ./cal_images --march nash-m
```

Recommended output naming:

- S100: `*_nashe_*_nv12.hbm`
- S100P: `*_nashm_*_nv12.hbm`

Place converted models in `model/nash-e/` or `model/nash-m/` so that `runtime/python/run.sh` and `runtime/python/main.py` can use them directly.

### 4. Script Arguments

```bash
python3 mapper.py -h
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--onnx` | Floating-point ONNX model path. | `./yolo11n.onnx` |
| `--output-dir` | Output directory for converted models. | `.` |
| `--cal-images` | Calibration image directory. | `./cal_images` |
| `--march` | Target architecture: `nash-e` for RDK S100, `nash-m` for RDK S100P. | `nash-e` |
| `--quantized` | Quantization precision: `int8` or `int16`. | `int8` |
| `--jobs` | Number of parallel compilation jobs. | `16` |
| `--optimize-level` | Compiler optimization level. Nash supports `O0` to `O2`. | `O2` |
| `--cal-sample` | Whether to sample images from the calibration directory. | `True` |
| `--cal-sample-num` | Number of sampled calibration images. | `20` |
| `--save-cache` | Whether to keep temporary workspace directories. | `False` |

## Input and Output Protocol

### Input Protocol

YOLO26 runtime uses NV12 input with two fixed input tensors:

- `input[0]`: Y plane
- `input[1]`: UV plane

Converted models must keep this input protocol.

### Output Protocol

The Python runtime parses outputs by fixed indexes:

- Detection: `[cls, box] * 3`
- Segmentation: `[cls, box, mask_coeff] * 3 + proto`
- Pose: `[cls, box, keypoints] * 3`
- OBB: `[cls, box, angle] * 3`
- Classification: one classification output tensor

See `runtime/python/yolo26_*.py` for the exact post-processing implementation.

## Compile Result Check

```bash
hrt_model_exec model_info --model_file yolo26n_detect_nashm_640x640_nv12.hbm
hrt_model_exec perf --model_file yolo26n_detect_nashm_640x640_nv12.hbm --thread_num 1
```

## Troubleshooting

- **Permission issues**: If copied files on the host have unexpected ownership, check file ownership or run `sudo chown -R`.
- **Memory or IPC errors**: Start Docker with `--shm-size=15g`.
- **Optimization-level errors**: If `O3` is not supported on Nash, use `O0`, `O1`, or `O2`.

## License

Tools in this directory follow the [Apache 2.0 License](../../../../LICENSE).
