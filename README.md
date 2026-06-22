English | [简体中文](./README_cn.md)

# Model Zoo
## Repository Introduction

This repository is the official BPU model example and tool collection (Model Zoo) provided by D-Robotics, oriented towards AI model deployment and application development running on BPU (Brain Processing Unit), helping developers quickly get started with BPU and fast-track model inference workflows.

The repository includes BPU-runnable models covering multiple AI domains (such as computer vision, speech, and embodied AI), and provides complete reference implementations from model preparation → inference execution → result parsing → example validation, helping users understand and utilize BPU capabilities at minimal cost.

Currently supported platforms:

- RDKS100
- RDKS100P

- RDKS600

Core Value

- 🚀 Quick BPU Adoption

    Provides ready-to-run models and example projects, helping users complete BPU inference validation in the shortest time.

- 🧩 Complete End-to-End Examples

    Covers model loading, preprocessing, BPU inference execution, postprocessing, and result visualization, supporting both C/C++ and Python interfaces.

- 📐 Standardized Design & Complete Interface Documentation

    Provides unified directory structures and sample code specifications, along with detailed interface documentation and usage instructions, making it easy for customers to quickly understand, perform secondary development, and reduce integration and maintenance costs.


## Repository Directory Structure

This repository adopts a layered, clearly defined directory structure to help users quickly locate the content they need and get started. The `samples/` directory is the core of the repository, providing various model examples running on BPU; `docs/` provides specification and interface documentation; `datasets/` stores datasets needed for examples and evaluation; `utils/` consolidates reusable common tools for batch maintenance.

The following shows the top-level directory structure for a quick overview:
```bash
.
|-- datasets                               # Public datasets and sample data
|-- docs                                   # Project documentation and user guides
|-- samples                                # Model examples (core content)
|-- tools                                  # Conversion/build/helper tools
|-- tros                                   # TROS/ROS related adapters
|-- utils                                  # Common utility library
|-- LICENSE                                # License
`-- README.md                              # Top-level readme

```

## Quick Start

Models in this repository are categorized by domain and summarized in the Model List below.
Users can quickly run a target model by following these steps:
- Find the target model in the model list based on your needs;
- Navigate to the corresponding model directory using the path provided in the table;
- After entering the model directory, read the README.md, which contains the model's functionality description, usage instructions, and complete run guide;


Example: YOLOv5
- Locate YOLOv5 in the model list below;
- Enter the model directory:

    ```bash
    cd samples/vision/yolov5
    ```
- Read the corresponding README documentation

- Follow the model's instructions to complete the inference example run and validation.


For a systematic understanding of the repository's overall structure, BPU usage, and interface capabilities, refer to:
```bash
docs/Model_Zoo_User_Guide.md
```

## Model List

The following table categorizes the models currently available in this repository by application domain for quick lookup.
For detailed descriptions, usage, and examples of each model, click the corresponding detail link to view the README.md in that model's directory.

| Category | Model Name | Model Path | Supported Platforms | Details |
|---|---|---|---|---|
| Vision Multi-task | Ultralytics YOLO (YOLOv5u / YOLOv8 / YOLOv10 / YOLO11 / YOLO12) | samples/vision/ultralytics_yolo | S100 / S100P | [README](samples/vision/ultralytics_yolo/README.md) |
| Vision Multi-task | YOLO26 | samples/vision/ultralytics_yolo26 | S100 / S100P | [README](samples/vision/ultralytics_yolo26/README.md) |
| Object Detection | YOLOv5x | samples/vision/yolov5 | S100 / S600 | [README](samples/vision/yolov5/README.md) |
| Object Detection | YOLO11 | samples/vision/yolo11 | S100 / S600 | [README](samples/vision/yolo11/README.md) |
| Object Detection | YOLOv13 (iMoonLab) | samples/vision/yolov13_imoonlab | S100 | [README](samples/vision/yolov13_imoonlab/README.md) |
| Multi-Object Tracking | ByteTrack | samples/vision/bytetrack | S100 / S100P / S600 | [README](samples/vision/bytetrack/README.md) |
| Instance Segmentation | YOLO11-Seg | samples/vision/yolo11_seg | S100 / S600 | [README](samples/vision/yolo11_seg/README.md) |
| Instance Segmentation | YOLOe11-Seg | samples/vision/yoloe11_seg | S100 | [README](samples/vision/yoloe11_seg/README.md) |
| Pose Estimation | YOLO11-Pose | samples/vision/yolo11_pose | S100 / S600 | [README](samples/vision/yolo11_pose/README.md) |
| Image Classification | ResNet18 | samples/vision/resnet18 | S100 / S600 | [README](samples/vision/resnet18/README.md) |
| Image Classification | ResNet50 | samples/vision/resnet50 | S100 | [README](samples/vision/resnet50/README.md) |
| Image Classification | ResNet152 | samples/vision/resnet152 | S100 | [README](samples/vision/resnet152/README.md) |
| Image Classification | MobileNetV1 | samples/vision/mobilenetv1 | S100 | [README](samples/vision/mobilenetv1/README.md) |
| Image Classification | MobileNetV2 | samples/vision/mobilenetv2 | S100 / S600 | [README](samples/vision/mobilenetv2/README.md) |
| Image Classification | MobileNetV3 | samples/vision/mobilenetv3 | S100 | [README](samples/vision/mobilenetv3/README.md) |
| Image Classification | MobileNetV4 | samples/vision/mobilenetv4 | S100 | [README](samples/vision/mobilenetv4/README.md) |
| Image Classification | EfficientNet-Lite | samples/vision/efficientnet | S100 | [README](samples/vision/efficientnet/README.md) |
| Image Classification | ViT | samples/vision/vit | S100 | [README](samples/vision/vit/README.md) |
| Image Classification | 3D ResNet (Video Action) | samples/vision/3dresnet | S100 | [README](samples/vision/3dresnet/README.md) |
| Semantic Segmentation | UnetMobileNet | samples/vision/unetmobilenet | S100 / S600 | [README](samples/vision/unetmobilenet/README.md) |
| Monocular Depth | Depth Anything V2 | samples/vision/depth_anything_v2 | S100 / S600 | [README](samples/vision/depth_anything_v2/README.md) |
| Vision Encoder | SigLIP | samples/vision/siglip | S100 / S100P | [README](samples/vision/siglip/README.md) |
| Point Cloud Segmentation | PointNet | samples/vision/pointnet | S100 | [README](samples/vision/pointnet/README.md) |
| Lane Detection | LaneNet | samples/vision/lanenet | S100 | [README](samples/vision/lanenet/README.md) |
| Text Recognition | PaddleOCR | samples/vision/paddle_ocr | S100 | [README](samples/vision/paddle_ocr/README.md) |
| Speech Recognition | ASR | samples/speech/asr | S100 / S600 | [README](samples/speech/asr/README.md) |
| Keyword Spotting | KWS | samples/speech/kws | S100 | [README](samples/speech/kws/README.md) |
| Embodied AI / Robot Policy | ACT (Action Chunking Transformer) | samples/vla/act | S100 | [README](samples/vla/act/README.md) |


## Documentation

- Each model's top-level directory contains a `README.md` with an overall introduction to the model. If you want to quickly learn about a specific model, navigate directly to the relevant directory;
- Each model has detailed interface descriptions. For code-level interface information, read the [source code documentation](docs/source_reference/README.md) and follow the instructions to build or browse the code documentation;
- To submit your own model sample, please carefully read the [rdk_model_zoo](docs/Model_Zoo_Repository_Guidelines.md) repository specifications;


## License
[Apache License 2.0](LICENSE)
