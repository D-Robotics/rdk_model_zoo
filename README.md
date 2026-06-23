<div align="center">
  <img src="docs/assets/model_zoo_logo.jpg" width="60%" alt="RDK Model Zoo Logo"/>
</div>

<div align="center">
  <h1 align="center">RDK Model Zoo — RDK S Series</h1>
  <p align="center">
    <b>Out-of-the-Box AI Model Deployment Pipelines and Full-Link Conversion Tutorials Based on D-Robotics BPU</b>
  </p>
</div>

<div align="center">

**English** | [简体中文](./README_cn.md)

<p align="center">
  <a href="https://github.com/D-Robotics/rdk_model_zoo/stargazers"><img src="https://img.shields.io/github/stars/D-Robotics/rdk_model_zoo?style=flat-square&logo=github&color=blue" alt="Stars"></a>
  <a href="https://github.com/D-Robotics/rdk_model_zoo/network/members"><img src="https://img.shields.io/github/forks/D-Robotics/rdk_model_zoo?style=flat-square&logo=github&color=blue" alt="Forks"></a>
  <a href="https://github.com/D-Robotics/rdk_model_zoo/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome"></a>
  <a href="https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s/LICENSE"><img src="https://img.shields.io/github/license/D-Robotics/rdk_model_zoo?style=flat-square" alt="License"></a>
  <a href="https://developer.d-robotics.cc"><img src="https://img.shields.io/badge/Community-D--Robotics-orange.svg?style=flat-square" alt="Community"></a>
</p>

</div>

## Introduction

> **Mission**: Dedicated to providing D-Robotics developers with extreme performance, out-of-the-box, and full-scenario AI deployment validation experiences.

This repository is the official collection of BPU model examples and tools (Model Zoo) provided by D-Robotics. It is oriented towards AI model deployment and application development on BPU (Brain Processing Unit), helping developers to **quickly get started with BPU** and **fast-track model inference workflows**.

The repository includes BPU-ready models across multiple AI domains and provides complete reference implementations from **Original Model (PyTorch/ONNX) → Fixed-point Quantization → Inference Execution → Result Parsing → Example Validation**, helping users understand and utilize BPU capabilities at minimal cost.

### Core Value

- 🚀 **Quick BPU Adoption**: Provides out-of-the-box inference pipelines to help users complete BPU inference validation and performance evaluation in the shortest time.
- 🧩 **Complete End-to-End Examples**: Covers the entire process from algorithm export and fixed-point quantization to efficient on-board execution (`.hbm`). Includes model loading, preprocessing, BPU inference execution, post-processing, and result visualization.
- 📐 **Standardized Design & Documentation**: Provides unified directory structures and sample code specifications, supporting Python (`hbm_runtime`) and C/C++ interfaces for easy understanding, secondary development, and reduced integration/maintenance costs.
- 🌐 **Full Scenario Coverage**: Covers classification, detection, segmentation, pose estimation, depth estimation, OCR, speech, and multi-modal models.

### Hardware & System Support

This repository uses hardware-specific branches to keep maintained samples, legacy demos, and board-specific documents clearly separated. The current `rdk_s` branch is the primary delivery branch for RDK S series boards (S100 / S100P / S600).

| Target Hardware | Branch | Description |
| :--- | :--- | :--- |
| RDK S series | [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s) | **Current branch.** Primary delivery branch for RDK S100, S100P, and S600. |
| RDK X5 | [`rdk_x5`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5) | Primary delivery branch for RDK X5. |
| RDK X3 | [`rdk_x3`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x3) | Branch for RDK X3 devices. |
| RDK S legacy demos | [RDK Model Zoo S](https://github.com/D-Robotics/rdk_model_zoo_s) | Historical archived demos for RDK S series boards. |

---

## Directory Structure

<details>
<summary><b>Click to expand project directory architecture</b></summary>

<br>

```bash
rdk_model_zoo/                       # rdk_s branch
|-- samples/
|   |-- vision/
|   |   |-- ultralytics_yolo/        # Detection / Segmentation / Pose / Classification
|   |   |-- ultralytics_yolo26/      # Detection / Segmentation / Pose / OBB / Classification
|   |   |-- yolov5/                  # Object detection
|   |   |-- yolo11/                  # Object detection
|   |   |-- yolo11_seg/              # Instance segmentation
|   |   |-- yolo11_pose/             # Pose estimation
|   |   |-- yoloe11_seg/             # Instance segmentation (prompt-free)
|   |   |-- yolov13_imoonlab/        # Object detection
|   |   |-- bytetrack/               # Multi-object tracking
|   |   |-- resnet18/                # Image classification
|   |   |-- resnet50/                # Image classification
|   |   |-- resnet152/               # Image classification
|   |   |-- mobilenetv1/             # Image classification
|   |   |-- mobilenetv2/             # Image classification
|   |   |-- mobilenetv3/             # Image classification
|   |   |-- mobilenetv4/             # Image classification
|   |   |-- efficientnet/            # Image classification
|   |   |-- vit/                     # Image classification
|   |   |-- 3dresnet/                # Video action classification
|   |   |-- unetmobilenet/           # Semantic segmentation
|   |   |-- depth_anything_v2/       # Monocular depth estimation
|   |   |-- siglip/                  # Vision encoder for VLM / VLA
|   |   |-- pointnet/                # Point cloud part segmentation
|   |   |-- lanenet/                 # Lane detection
|   |   `-- paddle_ocr/             # OCR text detection and recognition
|   |-- speech/
|   |   |-- asr/                     # Automatic speech recognition
|   |   `-- kws/                    # Keyword spotting
|   `-- vla/
|       `-- act/                    # Action Chunking Transformer (robot policy)
|-- docs/                            # Project guidelines and reference documentation
|-- datasets/                        # Sample datasets and download scripts
|-- tros/                            # TROS integration guides and examples
|-- utils/                           # Shared Python utilities
```

</details>

---

## Quick Start

1. **Check system version**: Ensure the target board is running a supported RDK OS.
2. **Connect hardware**: Ensure your RDK S board is powered and network-connected. SSH or VSCode Remote SSH is recommended.
3. **Read the model README first**: Always open the target directory `README.md` before running commands.
4. **Run a sample** (example: YOLOv5 on RDK S100):

```bash
cd samples/vision/yolov5/runtime/python
bash run.sh
```

The `run.sh` script automatically downloads the model, installs dependencies, and runs inference. Output images are saved in the current directory.

---

## Model List

| Category | Model Name | Model Path | Supported Platforms | Details |
| :--- | :--- | :--- | :--- | :---: |
| Vision Multi-task | Ultralytics YOLO (YOLOv5u / YOLOv8 / YOLOv9 / YOLOv10 / YOLO11 / YOLO12) | `samples/vision/ultralytics_yolo` | S100 / S100P / S600 | [Details](./samples/vision/ultralytics_yolo) |
| Vision Multi-task | YOLO26 | `samples/vision/ultralytics_yolo26` | S100 / S100P / S600 | [Details](./samples/vision/ultralytics_yolo26) |
| Object Detection | YOLOv5x | `samples/vision/yolov5` | S100 / S600 | [Details](./samples/vision/yolov5) |
| Object Detection | YOLO11 | `samples/vision/yolo11` | S100 / S600 | [Details](./samples/vision/yolo11) |
| Object Detection | YOLOv13 (iMoonLab) | `samples/vision/yolov13_imoonlab` | S100 | [Details](./samples/vision/yolov13_imoonlab) |
| Multi-Object Tracking | ByteTrack | `samples/vision/bytetrack` | S100 / S100P / S600 | [Details](./samples/vision/bytetrack) |
| Instance Segmentation | YOLO11-Seg | `samples/vision/yolo11_seg` | S100 / S600 | [Details](./samples/vision/yolo11_seg) |
| Instance Segmentation | YOLOe11-Seg (Prompt-Free) | `samples/vision/yoloe11_seg` | S100 | [Details](./samples/vision/yoloe11_seg) |
| Pose Estimation | YOLO11-Pose | `samples/vision/yolo11_pose` | S100 / S600 | [Details](./samples/vision/yolo11_pose) |
| Image Classification | ResNet18 | `samples/vision/resnet18` | S100 / S600 | [Details](./samples/vision/resnet18) |
| Image Classification | ResNet50 | `samples/vision/resnet50` | S100 / S600 | [Details](./samples/vision/resnet50) |
| Image Classification | ResNet152 | `samples/vision/resnet152` | S100 / S600 | [Details](./samples/vision/resnet152) |
| Image Classification | MobileNetV1 | `samples/vision/mobilenetv1` | S100 | [Details](./samples/vision/mobilenetv1) |
| Image Classification | MobileNetV2 | `samples/vision/mobilenetv2` | S100 / S600 | [Details](./samples/vision/mobilenetv2) |
| Image Classification | MobileNetV3 | `samples/vision/mobilenetv3` | S100 | [Details](./samples/vision/mobilenetv3) |
| Image Classification | MobileNetV4 | `samples/vision/mobilenetv4` | S100 | [Details](./samples/vision/mobilenetv4) |
| Image Classification | EfficientNet-Lite | `samples/vision/efficientnet` | S100 | [Details](./samples/vision/efficientnet) |
| Image Classification | ViT | `samples/vision/vit` | S100 | [Details](./samples/vision/vit) |
| Image Classification | 3D ResNet (Video Action) | `samples/vision/3dresnet` | S100 | [Details](./samples/vision/3dresnet) |
| Semantic Segmentation | UnetMobileNet | `samples/vision/unetmobilenet` | S100 / S600 | [Details](./samples/vision/unetmobilenet) |
| Monocular Depth Estimation | Depth Anything V2 | `samples/vision/depth_anything_v2` | S100 | [Details](./samples/vision/depth_anything_v2) |
| Vision Encoder | SigLIP | `samples/vision/siglip` | S100 / S100P | [Details](./samples/vision/siglip) |
| Point Cloud Segmentation | PointNet | `samples/vision/pointnet` | S100 | [Details](./samples/vision/pointnet) |
| Lane Detection | LaneNet | `samples/vision/lanenet` | S100 | [Details](./samples/vision/lanenet) |
| Text Recognition | PaddleOCR | `samples/vision/paddle_ocr` | S100 | [Details](./samples/vision/paddle_ocr) |
| Speech Recognition | ASR (Wav2Vec2) | `samples/speech/asr` | S100 / S600 | [Details](./samples/speech/asr) |
| Keyword Spotting | KWS (MDTC) | `samples/speech/kws` | S100 | [Details](./samples/speech/kws) |
| Embodied AI / Robot Policy | ACT (Action Chunking Transformer) | `samples/vla/act` | S100 / S600 | [Details](https://github.com/D-Robotics/rdk_LeRobot_tools) |

---

## Documentation & Resources

- **Model Docs**: Each model's top-level `README.md` provides an overview, run guide, and interface description.
- **Source Reference**: For code-level interface details, see **[Source Documentation](./docs/source_reference/README.md)**.
- **Guidelines**: To contribute or develop, please read the **[Model Zoo Repository Guidelines](./docs/Model_Zoo_Repository_Guidelines.md)**.
- **BPU Python API**: See **[Python API User Guide](./docs/Python_API_User_Guide.md)** for `hbm_runtime` usage.
- **UCP Interface**: See **[UCP User Guide](./docs/UCP_User_Guide.md)** for `libdnn` / `libucp` interface details.
- **Toolchain Manual**: [RDK S Series OE Toolchain](https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/overview)
- **Developer Forum**: [D-Robotics Developer Community](https://developer.d-robotics.cc/)

---

## FAQ

<details>
<summary><b>1. Model accuracy doesn't meet expectations?</b></summary>
<br>

- Ensure OpenExplorer Docker and board-side `hbm_runtime` versions are up-to-date.
- Check if model export followed the operator replacement steps described in the model's `conversion/README.md`.
- Verify cosine similarity of each output node is >= 0.999 (minimum 0.99) during quantization validation.
- For Transformer-based models (e.g., ViT, Depth Anything V2, SigLIP), int16 quantization is recommended over int8.
</details>

<details>
<summary><b>2. Inference speed doesn't meet expectations?</b></summary>
<br>

- Python API performance is lower than C/C++. For maximum performance, use the C/C++ runtime.
- Benchmark data (pure forward) excludes pre/post-processing. Models with **NV12** input usually achieve peak BPU throughput.
- Ensure CPU/BPU frequency is locked to maximum performance mode:

```bash
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/bpu/bpu0/devfreq/28108000.bpu/governor"
```
</details>

<details>
<summary><b>3. How to fix quantization precision loss?</b></summary>
<br>

- Refer to the PTQ accuracy debugging section in the OE toolchain documentation.
- If INT8 loss is severe (e.g., Softmax-heavy Transformer models), switch to INT16 quantization via `set_all_nodes_int16` in the YAML config.
- For severe cases, consider Mixed Precision or QAT (Quantization-Aware Training).
</details>

<details>
<summary><b>4. Does the model use CPU during inference?</b></summary>
<br>

Yes. Non-quantizable or BPU-unsupported operators fall back to CPU. Even for pure BPU models, input/output quantization/dequantization nodes are executed by the CPU. Use `hrt_model_exec model_info` to inspect operator placement.
</details>

<details>
<summary><b>5. How to check which BPU platform my board uses?</b></summary>
<br>

```bash
cat /sys/class/boardinfo/soc_name
```

- `s100` → RDK S100, BPU is Nash-e (80 TOPS @ int8)
- `s100p` → RDK S100P, BPU is Nash-m (128 TOPS @ int8)
- `s600` → RDK S600, BPU is Nash-p
</details>

---

## Community & Contribution

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=D-Robotics/rdk_model_zoo&type=Date)](https://star-history.com/#D-Robotics/rdk_model_zoo&Date)

We warmly welcome contributions! Please raise an issue on [GitHub Issues](https://github.com/D-Robotics/rdk_model_zoo/issues) or discuss on the [Developer Community](https://developer.d-robotics.cc/).

## License

This project is licensed under the [Apache License 2.0](./LICENSE) agreement.
