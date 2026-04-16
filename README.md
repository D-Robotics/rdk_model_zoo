<div align="center">
  <p><b>⚠️ Note: This repository is currently undergoing migration and refactoring. Some features and documentation may be incomplete. Thank you for your patience.</b></p>
</div>

<div align="center">
  <img src="resource/imgs/model_zoo_logo.jpg" width="60%" alt="RDK Model Zoo Logo"/>
</div>

<div align="center">
  <h1 align="center">RDK Model Zoo</h1>
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
  <a href="https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5/LICENSE"><img src="https://img.shields.io/github/license/D-Robotics/rdk_model_zoo?style=flat-square" alt="License"></a>
  <a href="https://developer.d-robotics.cc"><img src="https://img.shields.io/badge/Community-D--Robotics-orange.svg?style=flat-square" alt="Community"></a>
</p>

</div>

## Introduction

> **Mission**: Dedicated to providing D-Robotics developers with extreme performance, out-of-the-box, and full-scenario AI deployment validation experiences.

This repository is the official collection of BPU model examples and tools (Model Zoo) provided by D-Robotics. It is oriented towards AI model deployment and application development on BPU (Brain Processing Unit), helping developers to **quickly get started with BPU** and **fast-track model inference workflows**.

The repository includes BPU-ready models across multiple AI domains and provides complete reference implementations from **Original Model (PyTorch/ONNX) -> Fixed-point Quantization -> Inference Execution -> Result Parsing -> Example Validation**, helping users understand and utilize BPU capabilities at minimal cost.

### Core Value
- 🚀 **Quick BPU Adoption**: Provides out-of-the-box inference pipelines to help users complete BPU inference validation and performance evaluation in the shortest time.
- 🧩 **Complete End-to-End Examples**: Covers the entire process from algorithm export and fixed-point quantization to efficient on-board execution (`.bin` / `.hbm`). Includes model loading, preprocessing, BPU inference execution, post-processing, and result visualization.
- 📐 **Standardized Design & Documentation**: Provides unified directory structures and sample code specifications, supporting Python (`hbm_runtime`) and C/C++ interfaces for easy understanding, secondary development, and reduced integration/maintenance costs.
- **🌐 Full Scenario Coverage**: Covers classification, detection, segmentation, pose estimation, OCR, and cutting-edge multi-modal models like LLM.

### Hardware & System Support
- **RDK X5 (Bayse-e)**: Recommended to use RDK OS >= 3.5.0 (Based on Ubuntu 22.04 aarch64, TROS-Humble).
- **RDK S100/S600**: Please refer to the dedicated repository [RDK Model Zoo S](https://github.com/d-Robotics/rdk_model_zoo_s).

---

## Directory Structure

This repository adopts a clear, layered, and task-oriented directory structure for quick navigation.

<details>
<summary><b>📂 Click to expand project directory architecture</b></summary>

<br>

```bash
rdk_model_zoo/
├── demos/                 # 🚀 Core Model Examples (Categorized by task)
│   ├── classification/    # Classification (MobileNet, ResNet, ConvNeXt...)
│   ├── detect/            # Detection (YOLOv5~v12, FCOS...)
│   ├── Seg/               # Segmentation (YOLO-Seg...)
│   ├── Pose/              # Pose Estimation
│   ├── OCR/               # Optical Character Recognition (PaddleOCR)
│   ├── llm/               # Large Language/Multi-modal Models (CLIP, YOLO-World)
│   └── tools/             # Batch testing and evaluation tools
├── docs/                  # 📖 Project guidelines and reference documentation
├── datasets/              # 🗂️ Sample datasets and download scripts
├── utils/                 # 🛠️ Universal C++/Python utility library (Pre/post-processing, viz, etc.)
└── resource/              # 🖼️ Static resources (Test images, Logo, etc.)
```
</details>

---

## Quick Start

Models in this repository are categorized by task and summarized in the **Model Zoo Matrix** below. Follow these steps to quickly run a model:

1. **Find Model**: Locate your desired model in the matrix below.
2. **Connect Hardware**: Ensure your RDK board is powered and network-connected. SSH or VSCode Remote SSH is recommended.
3. **Install Dependencies**: Run the following command on the RDK board terminal (pre-installed on RDK OS >= 3.5.0): `pip install hbm_runtime`
4. **Run Example**: Navigate to the model directory, **carefully read the `README.md` there**, and follow the instructions.

> **Example: YOLO11 Object Detection**
> ```bash
> # 1. Clone repository
> git clone https://github.com/D-Robotics/rdk_model_zoo.git
> cd rdk_model_zoo
> 
> # 2. Enter model directory and read its README
> cd demos/detect/YOLO11/YOLO11-Detect_YUV420SP
> 
> # 3. Run inference (Model will be downloaded automatically)
> python3 YOLO11_Detect_YUV420SP.py --model-path ./model/yolo11n_det_640x640_nv12.bin --test-img ./data/bus.jpg
> ```

**Inference Result:**
<div align="center">
  <img src="resource/imgs/demo_rdkx5_yolov10n_detect.jpg" width="80%" alt="Inference Result"/>
</div>

---

## Model Zoo Matrix

Categorized by **Task Type**. Click the `Code` link to view the detailed README and examples for each model.

| Task Type | Representative Models | Demo Code |
| :--- | :--- | :---: |
| **Classification** | MobileNet (V1-V4), EfficientNet, ConvNeXt, ResNet, FastViT (20+ models) | [Code](./demos/classification) |
| **Object Detection** | YOLOv5, YOLOv8, YOLOv10, YOLO11, YOLO12, FCOS, LPRNet | [Code](./demos/detect) |
| **Segmentation** | YOLOv8-Seg, YOLO11-Seg, YOLOE-11-Seg-Prompt-Free | [Code](./demos/Seg) |
| **Pose Estimation** | YOLO11-Pose | [Code](./demos/Pose) |
| **Large Models** | CLIP, YOLO-World | [Code](./demos/llm) |
| **OCR** | PaddleOCR | [Code](./demos/OCR) |
| **Vision Specifics** | MODNet (Portrait Matting) | [Code](./demos/Vision) |

*(Continuously updating... PRs for new models are welcome!)*

---

## Documentation & Resources

- **Model Docs**: Each model's top-level `README.md` provides an overview and run guide.
- **Source Reference**: For code-level interface details, see **[Source Documentation](./docs/source_reference/README.md)**.
- **Guidelines**: To contribute or develop, please read the **[Model Zoo Repository Guidelines](./docs/Model_Zoo_Repository_Guidelines.md)**.
- **Toolchain Manuals**:
  - [RDK X5 Toolchain Doc](https://developer.d-robotics.cc/api/v1/fileData/x5_doc-v126cn/index.html)
  - [RDK X3 Toolchain Doc](https://developer.d-robotics.cc/api/v1/fileData/horizon_xj3_open_explorer_cn_doc/index.html)
- **Developer Forum**: [D-Robotics Developer Community](https://developer.d-robotics.cc/)
- **User Manual**: [RDK User Manual](https://developer.d-robotics.cc/information)

---

## FAQ

<details>
<summary><b>1. Model accuracy doesn't meet expectations?</b></summary>
<br>

- Ensure OpenExplorer Docker and board-side `libdnn.so` versions are up-to-date.
- Check if model export followed the structure adjustments/operator replacements required in the model's README.
- Verify cosine similarity of each output node is >= 0.999 (minimum 0.99) during quantization validation.
</details>

<details>
<summary><b>2. Inference speed doesn't meet expectations?</b></summary>
<br>

- Python API performance is lower than C/C++. For maximum performance, use C/C++.
- Benchmark data (pure forward) excludes pre/post-processing. Models with **NV12** input usually achieve peak BPU throughput.
- Ensure CPU/BPU frequency is locked to maximum.
- Check for other resource-heavy processes.
</details>

<details>
<summary><b>3. How to fix quantization precision loss?</b></summary>
<br>

- Refer to the PTQ accuracy debugging section in the platform documentation.
- If INT8 loss is severe due to model characteristics, consider Mixed Precision or QAT (Quantization-Aware Training).
</details>

<details>
<summary><b>4. Error "Can't reshape 1354752 in (1,3,640,640)"?</b></summary>
<br>

Update the resolution in `preprocess.py` to match your ONNX model's input size. Delete old calibration data and re-run the calibration script.
</details>

<details>
<summary><b>5. mAP accuracy is lower than official results (e.g., Ultralytics)?</b></summary>
<br>

- Deployment uses fixed shape and INT8 quantization, unlike dynamic shape/float official tests.
- Slight implementation differences in evaluation scripts (e.g., `pycocotools`).
- NCHW-RGB to NV12 conversion adds minimal pixel-level loss.
</details>

<details>
<summary><b>6. Does the model use CPU during inference?</b></summary>
<br>

Yes. Non-quantizable or BPU-unsupported operators **fallback** to CPU. Even for pure BPU models, input/output quantization/dequantization nodes are executed by the CPU.
</details>

---

## Community & Contribution

### Star History
[![Star History Chart](https://api.star-history.com/svg?repos=D-Robotics/rdk_model_zoo&type=Date)](https://star-history.com/#D-Robotics/rdk_model_zoo&Date)

We warmly welcome contributions! Please raise an issue on [GitHub Issues](https://github.com/D-Robotics/rdk_model_zoo/issues) or discuss on the [Developer Community](https://developer.d-robotics.cc/).

## License

This project is licensed under the [Apache License 2.0](./LICENSE) agreement.
