English | [简体中文](./README_cn.md)

# YOLO26 Model Evaluation

This directory contains scripts for evaluating the accuracy of various YOLO26 task models. It supports direct execution on RDK hardware and outputs industry-standard metrics.

## Environment Setup

- **Python Environment**: Ensure Python 3 is installed on your RDK device.
- **Dependencies**:
  - `pycocotools`: Used for mAP calculation on COCO datasets (Detection, Segmentation, Pose).
    ```bash
    pip install pycocotools
    ```
  - Base libraries: `opencv-python`, `numpy`, etc.

## Dataset Preparation

Scripts read data from the `datasets/` directory by default. Ensure the paths are correct:
- **Detection / Segmentation / Pose**: [COCO val2017](../../../../datasets/coco/README.md)
- **Classification**: [ImageNet val](../../../../datasets/imagenet/README.md)

## Usage

This directory provides evaluation scripts for various tasks. Before running the scripts, please ensure that the environment and datasets are prepared as described above.

### 1. Object Detection

Use the `eval_yolo26_det.py` script to evaluate the detection model's mAP on the COCO dataset.

```bash
python3 eval_yolo26_det.py \
  --model-path ../model/nash-e/yolo26n_detect_nashe_640x640_nv12.hbm \
  --image-dir ../../../../datasets/coco/val2017 \
  --annotation ../../../../datasets/coco/annotations/instances_val2017.json \
  --conf-thres 0.25 \
  --nms-thres 0.7
```

**Arguments**:
- `--model-path`: Path to the quantized model file (.hbm).
- `--image-dir`: Directory containing validation images.
- `--annotation`: Path to the COCO format annotation file.
- `--conf-thres`: Confidence threshold (default 0.25; set lower for accuracy testing to get higher recall).
- `--nms-thres`: NMS threshold.

### 2. Image Classification

Use the `eval_yolo26_cls.py` script to evaluate the classification model's Top-1/Top-5 accuracy on the ImageNet dataset.

```bash
python3 eval_yolo26_cls.py \
  --model-path ../model/nash-e/yolo26n_cls_nashe_224x224_nv12.hbm \
  --image-dir ../../../../datasets/imagenet/val \
  --val-txt ../../../../datasets/imagenet/val.txt \
  --topk 5
```

**Arguments**:
- `--val-txt`: Text file containing filename to label mapping (format: `filename label_id`).

### 3. Instance Segmentation

Use the `eval_yolo26_seg.py` script to evaluate the segmentation model.

```bash
python3 eval_yolo26_seg.py \
  --model-path ../model/nash-e/yolo26n_seg_nashe_640x640_nv12.hbm \
  --image-dir ../../../../datasets/coco/val2017 \
  --annotation ../../../../datasets/coco/annotations/instances_val2017.json
```

### 4. Pose Estimation

Use the `eval_yolo26_pose.py` script to evaluate the pose estimation model.

```bash
python3 eval_yolo26_pose.py \
  --model-path ../model/nash-e/yolo26n_pose_nashe_640x640_nv12.hbm \
  --image-dir ../../../../datasets/coco/val2017 \
  --annotation ../../../../datasets/coco/annotations/person_keypoints_val2017.json
```

### 5. Oriented Bounding Box (OBB)

Use the `eval_yolo26_obb.py` script to evaluate the oriented object detection model (typically using DOTA dataset format).

```bash
python3 eval_yolo26_obb.py \
  --model-path ../model/nash-e/yolo26n_obb_nashe_640x640_nv12.hbm \
  --image-dir ../../../../datasets/dotav1/val \
```

## Benchmark Results

### RDK S100/S100P Performance Data (Performance @ NV12)

| Device | Model | Size <br> (Pixels) | Classes | BPU Task Latency / <br> BPU Throughput (Threads) | CPU Latency | Params <br> (M) | FLOPs <br> (G) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S100 | YOLO26n Detect | 640x640 | 80 | 1.70 ms / 499.33 FPS (1 thread) <br> 2.31 ms / 779.33 FPS (2 threads) | - | 2.57 | 6.1 |
| S100 | YOLO26s Detect | 640x640 | 80 | 2.87 ms / 314.84 FPS (1 thread) <br> 4.63 ms / 409.84 FPS (2 threads) | - | 10.01 | 22.8 |
| S100 | YOLO26m Detect | 640x640 | 80 | 6.06 ms / 157.14 FPS (1 thread) <br> 10.99 ms / 177.56 FPS (2 threads) | - | 21.90 | 75.4 |
| S100 | YOLO26l Detect | 640x640 | 80 | 7.35 ms / 130.56 FPS (1 thread) <br> 13.51 ms / 144.92 FPS (2 threads) | - | 26.30 | 93.8 |
| S100 | YOLO26x Detect | 640x640 | 80 | 13.91 ms / 70.33 FPS (1 thread) <br> 26.58 ms / 74.37 FPS (2 threads) | - | 58.99 | 209.5 |
| S100 | YOLO26n Seg | 640x640 | 80 | 2.23 ms / 354.15 FPS (1 thread) <br> 2.97 ms / 566.80 FPS (2 threads) | - | 3.13 | 10.5 |
| S100 | YOLO26s Seg | 640x640 | 80 | 3.80 ms / 234.05 FPS (1 thread) <br> 6.17 ms / 299.39 FPS (2 threads) | - | 11.51 | 37.4 |
| S100 | YOLO26m Seg | 640x640 | 80 | 9.08 ms / 103.15 FPS (1 thread) <br> 16.21 ms / 118.81 FPS (2 threads) | - | 27.11 | 132.5 |
| S100 | YOLO26l Seg | 640x640 | 80 | 10.48 ms / 90.13 FPS (1 thread) <br> 19.31 ms / 100.34 FPS (2 threads) | - | 31.52 | 150.9 |
| S100 | YOLO26x Seg | 640x640 | 80 | 19.90 ms / 48.70 FPS (1 thread) <br> 38.17 ms / 51.53 FPS (2 threads) | - | 70.69 | 337.7 |
| S100 | YOLO26n Pose | 640x640 | 1 | 1.81 ms / 486.37 FPS (1 thread) <br> 2.59 ms / 713.83 FPS (2 threads) | - | 3.68 | 10.3 |
| S100 | YOLO26s Pose | 640x640 | 1 | 3.07 ms / 300.62 FPS (1 thread) <br> 5.03 ms / 380.99 FPS (2 threads) | - | 11.81 | 29.2 |
| S100 | YOLO26m Pose | 640x640 | 1 | 6.42 ms / 149.73 FPS (1 thread) <br> 11.68 ms / 167.90 FPS (2 threads) | - | 24.22 | 85.2 |
| S100 | YOLO26l Pose | 640x640 | 1 | 7.75 ms / 124.84 FPS (1 thread) <br> 14.30 ms / 137.49 FPS (2 threads) | - | 28.63 | 103.6 |
| S100 | YOLO26x Pose | 640x640 | 1 | 14.45 ms / 67.87 FPS (1 thread) <br> 27.60 ms / 71.79 FPS (2 threads) | - | 62.73 | 225.3 |
| S100 | YOLO26n Cls | 224x224 | 1000 | 0.56 ms / 1659.68 FPS (1 thread) <br> 0.62 ms / 3112.09 FPS (2 threads) | - | 2.81 | 0.5 |
| S100 | YOLO26s Cls | 224x224 | 1000 | 0.74 ms / 1293.15 FPS (1 thread) <br> 0.83 ms / 2332.95 FPS (2 threads) | - | 6.72 | 1.6 |
| S100 | YOLO26m Cls | 224x224 | 1000 | 1.13 ms / 853.03 FPS (1 thread) <br> 1.56 ms / 1266.73 FPS (2 threads) | - | 11.63 | 5.0 |
| S100 | YOLO26l Cls | 224x224 | 1000 | 1.34 ms / 723.45 FPS (1 thread) <br> 1.97 ms / 999.78 FPS (2 threads) | - | 14.12 | 6.2 |
| S100 | YOLO26x Cls | 224x224 | 1000 | 2.08 ms / 471.88 FPS (1 thread) <br> 3.46 ms / 573.27 FPS (2 threads) | - | 29.64 | 13.7 |
| S100 | YOLO26n Obb | 640x640 | 15 | 1.64 ms / 548.83 FPS (1 thread) <br> 2.24 ms / 842.21 FPS (2 threads) | - | 2.65 | 6.3 |
| S100 | YOLO26s Obb | 640x640 | 15 | 2.80 ms / 337.46 FPS (1 thread) <br> 4.65 ms / 418.28 FPS (2 threads) | - | 10.53 | 24.5 |
| S100 | YOLO26m Obb | 640x640 | 15 | 6.16 ms / 157.61 FPS (1 thread) <br> 11.27 ms / 175.02 FPS (2 threads) | - | 23.49 | 82.2 |
| S100 | YOLO26l Obb | 640x640 | 15 | 7.46 ms / 130.74 FPS (1 thread) <br> 13.79 ms / 143.34 FPS (2 threads) | - | 27.90 | 100.6 |
| S100 | YOLO26x Obb | 640x640 | 15 | 13.81 ms / 71.33 FPS (1 thread) <br> 26.95 ms / 73.69 FPS (2 threads) | - | 62.66 | 225.3 |
| S100P | YOLO26n Detect | 640x640 | 80 | - | - | 2.57 | 6.1 |
| S100P | YOLO26s Detect | 640x640 | 80 | - | - | 10.01 | 22.8 |
| S100P | YOLO26m Detect | 640x640 | 80 | - | - | 21.90 | 75.4 |
| S100P | YOLO26l Detect | 640x640 | 80 | - | - | 26.30 | 93.8 |
| S100P | YOLO26x Detect | 640x640 | 80 | - | - | 58.99 | 209.5 |

### RDK S100 Accuracy Data (Accuracy @ NV12 - Detection)

| Device | Model | Accuracy bbox-all <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-small <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-medium <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-large <br> mAP @.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S100 | YOLO26n Detect | 0.319 / 0.286 (89.7 %) | 0.107 / 0.083 (77.6 %) | 0.349 / 0.304 (87.1 %) | 0.508 / 0.473 (93.1 %) |
| S100 | YOLO26s Detect | 0.395 / 0.362 (91.6 %) | 0.183 / 0.163 (89.1 %) | 0.440 / 0.402 (91.4 %) | 0.583 / 0.524 (89.9 %) |
| S100 | YOLO26m Detect | 0.442 / 0.413 (93.4 %) | 0.242 / 0.202 (83.5 %) | 0.489 / 0.456 (93.3 %) | 0.629 / 0.603 (95.9 %) |
| S100 | YOLO26l Detect | 0.456 / 0.440 (96.5 %) | 0.260 / 0.230 (88.5 %) | 0.499 / 0.489 (98.0 %) | 0.627 / 0.623 (99.4 %) |
| S100 | YOLO26x Detect | 0.484 / 0.449 (92.8 %) | 0.292 / 0.246 (84.2 %) | 0.528 / 0.488 (92.4 %) | 0.669 / 0.646 (96.6 %) |

### RDK S100P Accuracy Data (Accuracy @ RGB - Detection)

| Device | Model | Accuracy bbox-all <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-small <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-medium <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-large <br> mAP @.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S100P | YOLO26n Detect | 0.319 / 0.290 (91.0 %) | 0.107 / 0.087 (81.7 %) | 0.349 / 0.313 (89.8 %) | 0.508 / 0.463 (91.1 %) |
| S100P | YOLO26s Detect | 0.395 / 0.367 (93.0 %) | 0.183 / 0.174 (94.8 %) | 0.440 / 0.410 (93.1 %) | 0.583 / 0.530 (91.0 %) |
| S100P | YOLO26m Detect | 0.442 / 0.421 (95.2 %) | 0.242 / 0.224 (92.6 %) | 0.489 / 0.460 (94.0 %) | 0.629 / 0.603 (95.8 %) |
| S100P | YOLO26l Detect | 0.456 / 0.437 (96.0 %) | 0.260 / 0.234 (89.8 %) | 0.499 / 0.484 (97.1 %) | 0.627 / 0.609 (97.0 %) |
| S100P | YOLO26x Detect | 0.484 / 0.466 (96.2 %) | 0.292 / 0.271 (92.8 %) | 0.528 / 0.502 (95.1 %) | 0.669 / 0.654 (97.8 %) |

### RDK S100 Accuracy Data (Accuracy @ NV12 - Pose Estimation)

| Device | Model | Accuracy kpt-all <br> mAP @.50:.95 <br> (BPU Python) | Accuracy kpt-medium <br> mAP @.50:.95 <br> (BPU Python) | Accuracy kpt-large <br> mAP @.50:.95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- | :--- |
| S100 | YOLO26n Pose | 0.504 | 0.412 | 0.647 |
| S100 | YOLO26s Pose | 0.575 | 0.498 | 0.697 |
| S100 | YOLO26m Pose | 0.620 | 0.554 | 0.737 |
| S100 | YOLO26l Pose | 0.646 | 0.579 | 0.744 |
| S100 | YOLO26x Pose | 0.663 | 0.601 | 0.775 |

### RDK S100P Accuracy Data (Accuracy @ NV12 - Pose Estimation)

| Device | Model | Accuracy kpt-all <br> mAP @.50:.95 <br> (BPU Python) | Accuracy kpt-medium <br> mAP @.50:.95 <br> (BPU Python) | Accuracy kpt-large <br> mAP @.50:.95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- | :--- |
| S100P | YOLO26n Pose | - | - | - |

### RDK S100 Accuracy Data (Accuracy @ NV12 - Segmentation)

| Device | Model | Accuracy mask-all <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-small <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-medium <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-large <br> mAP @.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S100 | YOLO26n Seg | - / 0.254 | - / 0.057 | - / 0.269 | - / 0.434 |
| S100 | YOLO26s Seg | - / 0.330 | - / 0.119 | - / 0.367 | - / 0.510 |
| S100 | YOLO26m Seg | - / 0.356 | - / 0.148 | - / 0.399 | - / 0.536 |
| S100 | YOLO26l Seg | - / 0.375 | - / 0.164 | - / 0.419 | - / 0.560 |
| S100 | YOLO26x Seg | - / 0.381 | - / 0.176 | - / 0.426 | - / 0.576 |

### RDK S100P Accuracy Data (Accuracy @ NV12 - Segmentation)

| Device | Model | Accuracy mask-all <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-small <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-medium <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-large <br> mAP @.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S100P | YOLO26n Seg | - / - (- %) | - / - (- %) | - / - (- %) | - / - (- %) |

### RDK S100 Accuracy Data (Accuracy @ NV12 - Classification)

| Device | Model | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :--- | :--- | :--- |
| S100 | YOLO26n Cls | 0.6165 | 0.8359 |
| S100 | YOLO26s Cls | 0.6854 | 0.8853 |
| S100 | YOLO26m Cls | 0.7194 | 0.9080 |
| S100 | YOLO26l Cls | 0.7369 | 0.9168 |
| S100 | YOLO26x Cls | 0.7432 | 0.9222 |

### RDK S100P Accuracy Data (Accuracy @ NV12 - Classification)

| Device | Model | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :--- | :--- | :--- |
| S100P | YOLO26n Cls | - | - |

### RDK S100 Accuracy Data (Accuracy @ NV12 - OBB)

| Device | Model | Accuracy mAP50 <br> (BPU Python) | Accuracy mAP50-95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- |
| S100 | YOLO26n Obb | - | - |
| S100 | YOLO26s Obb | - | - |
| S100 | YOLO26m Obb | - | - |
| S100 | YOLO26l Obb | - | - |
| S100 | YOLO26x Obb | - | - |

### RDK S100P Accuracy Data (Accuracy @ NV12 - OBB)

| Device | Model | Accuracy mAP50 <br> (BPU Python) | Accuracy mAP50-95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- |
| S100P | YOLO26n Obb | - | - |


## Output Metrics

- **Detection / Segmentation / Pose**: Outputs AP @ IoU=0.50:0.95 (all, small, medium, large), AP @ 0.5, AP @ 0.75, and Recall metrics.
- **Classification**: Outputs Top-1 Accuracy, Top-5 Accuracy, and Inference FPS.
- **OBB**: Outputs mAP50 and mAP50-95 (DOTA metric).

## Performance Test Instructions

- **Device**: The hardware device used for testing, such as S100, S100P, etc.
- **Model**: The model used for testing, such as YOLO26n, YOLO26s, etc.
- **Classes**: The number of detection classes, consistent with COCO2017 or ImageNet-1k datasets.
- **BPU Task Latency / Throughput**:
  - **Single-thread Latency**: The ideal inference latency for a single frame on a single thread and single BPU core.
  - **Multi-thread Throughput**: The frames per second (FPS) achieved when submitting tasks to the BPU with multiple threads simultaneously. Usually, 2 threads provide a good balance between low latency and high BPU utilization.
  - **Test Command**: Uses the `hrt_model_exec` tool from the development kit: `hrt_model_exec perf --thread_num 2 --model_file <model.hbm>`. This tool measures the time from task submission to completion and accounts for cache warmup.
- **CPU Latency (Single Core)**: Post-processing time. It is positively correlated with the number of detected objects (data in this table is based on scenarios with < 100 objects). Both Python and C++ implementations are deeply optimized.
- **Memory Management**: In streaming inference, input/output memory should be allocated once and reused. Latency measurements should not include memory allocation and deallocation time.
- **Params (M) & FLOPs (G)**: Parameter count and computation volume of the original FP32 model (derived from Ultralytics YOLO export logs). Due to compiler optimizations, the actual BPU computation volume may differ.

## Accuracy Test Instructions

- **Device and Model Columns**: These carry the same definitions as described in the Performance Test Instructions section.
- **Calculation Tool**: Accuracy data is calculated using the official, unmodified `pycocotools` library from Microsoft.
- **Evaluation Modes**:
  - **Object Detection**: `iouType="bbox"`
  - **Instance Segmentation**: `iouType="bbox"` and `iouType="segm"`
  - **Human Pose Estimation**: `iouType="keypoints"`

- **Metric Definitions**:
  - `Accuracy bbox-all mAP @.50:.95`: Derived from `Average Precision (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ]`.
  - `Accuracy bbox-small/medium/large`: Represents AP for objects of different scales as defined by COCO.

- **AP vs. AR**: AP (Average Precision) focuses on "Quality" (both high Recall and high Precision with accurate localization). AR (Average Recall) focuses on "Quantity" (finding targets regardless of false positives). This evaluation uniformly uses **AP metrics** to measure model accuracy.

- **Test Procedure**: Inference is performed on 5,000 images from the COCO2017 validation set directly on the board. Results are dumped to a JSON file and processed by `pycocotools`.
  - **Score Threshold**: 0.25
  - **NMS Threshold**: 0.7

- **Accuracy Discrepancy Notes**:
  - Metrics calculated by `pycocotools` are typically slightly lower than those from `ultralytics` official tools. This is because `pycocotools` uses rectangular integration for the area under the AP curve, while `ultralytics` uses trapezoidal integration. Our primary focus is the comparison between fixed-point (quantized) and floating-point models using the same methodology to assess quantization loss.
  - **Classification**: Evaluated on the ImageNet-1k dataset using Top-1 and Top-5 accuracy.
  - **Color Space Conversion**: The BPU model introduces minor precision loss when converting NCHW-RGB888 input to YUV420SP (NV12). Incorporating color space conversion loss during training can mitigate this.
  - **Interface Variance**: Minor precision differences may occur between Python and C/C++ interfaces due to different handling of floating-point numbers during `memcpy` and data conversion.

- **Quantization Details**: The data in this report is based on **PTQ (Post-Training Quantization)** using 50 images for calibration. This is intended to simulate the out-of-the-box accuracy a developer would experience upon first compilation. It does not involve deep accuracy tuning or QAT (Quantization-Aware Training) and does not represent the theoretical upper bound of the model's precision.
