# YOLO26 Model Evaluation

This directory contains scripts for evaluating the accuracy of YOLO26 models across various tasks on D-Robotics RDK hardware.

## Prerequisites

- **Python Environment**: Ensure Python 3 is installed on your RDK board.
- **Dependencies**:
  - `pycocotools`: Required for mAP calculation on COCO datasets (Detection, Segmentation, Pose).
    ```bash
    pip install pycocotools
    ```
  - `opencv-python`, `numpy`, and other standard libraries.

## Dataset Preparation

The scripts default to reading data from the `datasets/` directory. Ensure your datasets are placed correctly:
- **Detection / Segmentation / Pose**: [COCO val2017](../../../../datasets/coco/README.md)
- **Classification**: [ImageNet val](../../../../datasets/imagenet/README.md)
- **OBB**: [DOTA val](../../../../datasets/dotav1/README.md)

## Usage

### 1. Object Detection Evaluation
Run `eval_yolo26_det.py` to calculate COCO mAP:
```bash
python3 eval_yolo26_det.py --model-path ../model/yolo26n_det.bin --limit 100
```

### 2. Instance Segmentation Evaluation
Run `eval_yolo26_seg.py` to calculate Mask mAP:
```bash
python3 eval_yolo26_seg.py --model-path ../model/yolo26n_seg.bin --limit 100
```

### 3. Pose Estimation Evaluation
Run `eval_yolo26_pose.py` to calculate Keypoints mAP:
```bash
python3 eval_yolo26_pose.py --model-path ../model/yolo26n_pose.bin --limit 100
```

### 4. Image Classification Evaluation
Run `eval_yolo26_cls.py` to calculate Top-1 and Top-5 accuracy:
```bash
python3 eval_yolo26_cls.py --model-path ../model/yolo26n_cls.bin --image-path /path/to/imagenet/val --val-txt /path/to/val.txt
```

### 5. Oriented Bounding Box (OBB) Evaluation
Run `eval_yolo26_obb.py` to generate 8-point polygon predictions:
```bash
python3 eval_yolo26_obb.py --model-path ../model/yolo26n_obb.bin --limit 100
```

## Argument Descriptions

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--model-path` | Path to the BPU quantized model (.bin) | Required |
| `--image-path` | Path to the validation image directory | Task-specific default |
| `--ann-path` | Path to the official annotation JSON file | Task-specific default |
| `--val-txt` | Ground truth text file (for Classification) | None |
| `--json-save-path` | Path to save prediction results (.json) | yolo26_xxx_results.json |
| `--conf-thres` | Confidence threshold | 0.25 |
| `--nms-thres` | Non-Maximum Suppression threshold | 0.7 (0.2 for OBB) |
| `--limit` | Limit the number of images to process (0 for all) | 0 |

## Benchmark Results

### RDK X5 Performance Data

| Device | Model | Size <br> (Pixels) | Classes | BPU Task Latency / <br> BPU Throughput (Threads) | CPU Latency | params <br> (M) | FLOPs <br> (B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| X5 | YOLO26n Detect | 640x640 | 80 | 11.6 ms / 86.3 FPS (1 thread) <br> 19.1 ms / 104.3 FPS (2 threads) | - | - | - |
| X5 | YOLO26s Detect | 640x640 | 80 | 20.9 ms / 47.7 FPS (1 thread) <br> 37.8 ms / 52.8 FPS (2 threads) | - | - | - |
| X5 | YOLO26m Detect | 640x640 | 80 | 51.1 ms / 24.8 FPS (1 thread) <br> 76.1 ms / 26.1 FPS (2 threads) | - | - | - |
| X5 | YOLO26l Detect | 640x640 | 80 | 40.1 ms / 19.5 FPS (1 thread) <br> 98.0 ms / 20.3 FPS (2 threads) | - | - | - |
| X5 | YOLO26x Detect | 640x640 | 80 | 103.3 ms / 9.6 FPS (1 thread) <br> 202.0 ms / 9.8 FPS (2 threads) | - | - | - |
| X5 | YOLO26n Seg | 640x640 | 80 | 15.5 ms / 64.3 FPS (1 thread) <br> 22.8 ms / 87.6 FPS (2 threads) | - | - | - |
| X5 | YOLO26n Pose | 640x640 | 80 | 12.5 ms / 79.6 FPS (1 thread) <br> 20.1 ms / 98.7 FPS (2 threads) | - | - | - |
| X5 | YOLO26n Cls | 224x224 | 1000 | 1.1 ms / 906.0 FPS (1 thread) <br> 1.7 ms / 1156.8 FPS (2 threads) | - | - | - |

### RDK X5 Accuracy Data (Accuracy @ NV12 - Detection)

| Device | Model | Accuracy bbox-all <br> mAP@.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-small <br> mAP@.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-medium <br> mAP@.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-large <br> mAP@.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| X5 | YOLO26n Detect | 0.319 / 0.284 (89.0 %) | 0.107 / 0.075 (70.1 %) | 0.349 / 0.299 (85.7 %) | 0.508 / 0.467 (91.9 %) |
| X5 | YOLO26s Detect | 0.395 / 0.357 (90.4 %) | 0.183 / 0.154 (84.2 %) | 0.440 / 0.393 (89.3 %) | 0.583 / 0.534 (91.6 %) |
| X5 | YOLO26m Detect | 0.442 / 0.413 (93.4 %) | 0.242 / 0.206 (85.1 %) | 0.489 / 0.454 (92.8 %) | 0.629 / 0.605 (96.1 %) |
| X5 | YOLO26l Detect | 0.456 / 0.431 (94.5 %) | 0.260 / 0.215 (82.7 %) | 0.499 / 0.479 (96.0 %) | 0.627 / 0.618 (98.6 %) |
| X5 | YOLO26x Detect | 0.484 / 0.438 (90.5 %) | 0.292 / 0.230 (78.8 %) | 0.528 / 0.479 (90.7 %) | 0.669 / 0.635 (94.9 %) |

### RDK X5 Accuracy Data (Accuracy @ NV12 - Segmentation)

| Device | Model | Accuracy mask-all <br> mAP@.50:.95 <br> (BPU Python) | Accuracy mask-small <br> mAP@.50:.95 <br> (BPU Python) | Accuracy mask-medium <br> mAP@.50:.95 <br> (BPU Python) | Accuracy mask-large <br> mAP@.50:.95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| X5 | YOLO26n Seg | 0.285 | 0.090 | 0.307 | 0.464 |

### RDK X5 Accuracy Data (Accuracy @ NV12 - Pose Estimation)

| Device | Model | Accuracy kpt-all <br> mAP@.50:.95 <br> (BPU Python) | Accuracy kpt-medium <br> mAP@.50:.95 <br> (BPU Python) | Accuracy kpt-large <br> mAP@.50:.95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- | :--- |
| X5 | YOLO26n Pose | 0.498 | 0.404 | 0.647 |

## Performance Test Instructions

- **Device**: Test platform. `S100P` for RDK S100P, `S100` for RDK S100, and `X5` for RDK X5 (Module).
- **Model**: The model under test, corresponding to the Support Models section.
- **Size (Pixels)**: The algorithm resolution (input resolution when exporting ONNX). Images of other resolutions are generally scaled to this size before inference.
- **Classes**: Number of detected classes, consistent with COCO2017 or ImageNet-1k datasets.
- **BPU Task Latency / Throughput**:
  - **Single-thread Latency**: The ideal latency for a single frame, single thread, and single BPU core.
  - **Multi-thread Throughput**: FPS achieved when multiple threads submit tasks to the BPU simultaneously. Typically, 2 threads balance low latency and high BPU utilization.
  - **Test Command**: Uses `hrt_model_exec` from the OE package: `hrt_model_exec perf --thread_num 2 --model_file <model.bin>`. This measures the time from task submission to completion, accounting for cache warmup.
- **CPU Latency (Single Core)**: Post-processing time. Positively correlated with the number of detected objects (data based on < 100 objects). Python and C++ implementations are both highly optimized.
- **Memory Management**: In streaming inference, input/output memory should be allocated once and reused. Do not include allocation/deallocation time in latency measurements.
- **Params (M) & FLOPs (B)**: Parameters and computation of the original FP32 model (from Ultralytics YOLO export logs). Actual BPU computation may vary due to compiler optimizations.

## Accuracy Test Instructions

- **Calculation**: Accuracy is calculated using the official, unmodified `pycocotools` library.
- **Evaluation Modes**:
  - Object Detection: `iouType="bbox"`
  - Instance Segmentation: `iouType="bbox"` and `iouType="segm"`
  - Pose Estimation: `iouType="keypoints"`
- **Metric Definitions**:
  - `Accuracy bbox-all mAP @.50:.95` is from `Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ]`.
  - Similar definitions apply to `small`, `medium`, and `large` areas.
- **AP vs AR**: AP focuses on "Quality" (finding targets correctly with precise boxes and categories); AR focuses on "Quantity" (finding as many targets as possible without penalizing false positives). AP is used as the primary metric here.
- **Test Data**: 5000 images from the COCO2017 validation set. Inference is performed on-board, results are dumped to JSON and evaluated via `pycocotools`. Defaults: `score_thres=0.25`, `nms_thres=0.7`.
- **Note on Accuracy Differences**:
  - `pycocotools` metrics may be slightly lower than `ultralytics` results because `pycocotools` uses rectangular integration while `ultralytics` uses trapezoidal integration for the AP curve. We focus on consistent comparison between fixed-point and floating-point models to assess quantization loss.
  - **Classification**: Evaluated on ImageNet-1k using Top-1 and Top-5 accuracy.
  - **Color Space**: Converting NCHW-RGB888 input to YUV420SP (NV12) for BPU may introduce minor loss. This can be mitigated by accounting for color conversion during training.
  - **Language Interface**: Minor differences between Python and C++ results may occur due to floating-point handling during `memcpy` and data structure conversion.
- **Quantization**: Results are based on **PTQ (Post-Training Quantization)** with 50 calibration images. This simulates a "first-time" compilation experience without intensive tuning or QAT, representing standard validation rather than the maximum possible accuracy.
