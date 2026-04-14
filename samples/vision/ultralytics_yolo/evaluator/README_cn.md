[English](./README.md) | [简体中文](./README_cn.md)

# Ultralytics YOLO 模型评测

本目录提供 Ultralytics YOLO 模型在 D-Robotics RDK X5 平台上的评测脚本和 benchmark 参考数据。

## 评测前提

- **Python 环境**：请确认开发板已安装 Python 3。
- **依赖库**：
  - `pycocotools`：用于检测、分割、姿态任务的 COCO mAP 计算。
    ```bash
    pip install pycocotools
    ```
  - `opencv-python`、`numpy` 以及脚本依赖的其他标准 Python 库。

## 数据集准备

评测脚本默认使用以下数据集路径：

- **检测 / 分割 / 姿态**：[COCO2017 val](../../../../resource/datasets/COCO2017/README.md)
- **分类**：[ImageNet val](../../../../resource/datasets/ImageNet/README.md)

请在运行脚本前确认数据集已经按默认目录准备完成。

## 使用方法

### 1. 目标检测评测

运行 `eval_Ultralytics_YOLO_Detect_YUV420SP.py`，导出 COCO 格式的检测结果 JSON：

```bash
python3 eval_Ultralytics_YOLO_Detect_YUV420SP.py \
  --model-path source/reference_bin_models/det/yolo12n_detect_bayes-e_640x640_nv12.bin \
  --image-path ../../../resource/datasets/COCO2017/val2017 \
  --json-path yolo12n_detect_bayes-e_640x640_nv12_py_coco2017_val_pridect.json
```

### 2. 实例分割评测

运行 `eval_Ultralytics_YOLO_Seg_YUV420SP.py`，导出 COCO 格式的检测和分割结果：

```bash
python3 eval_Ultralytics_YOLO_Seg_YUV420SP.py \
  --model-path source/reference_bin_models/seg/yolo11n-seg_detect_bayes-e_640x640_nv12.bin \
  --image-path ../../../resource/datasets/COCO2017/val2017 \
  --json-path yolo11n_seg2_bayese_640x640_nv12_coco2017_val_pridect_0_5.json
```

### 3. 姿态估计评测

运行 `eval_Ultralytics_YOLO_Pose_YUV420SP.py`，导出 COCO 格式的关键点结果：

```bash
python3 eval_Ultralytics_YOLO_Pose_YUV420SP.py \
  --model-path source/reference_bin_models/pose/yolov8n-pose_detect_bayes-e_640x640_nv12.bin \
  --image-path ../../../resource/datasets/COCO2017/val2017 \
  --json-path yolov8n-pose_detect_bayes-e_640x640_nv12_py_coco2017_val_pridect.json
```

### 4. 图像分类评测

运行 `eval_Ultralytics_YOLO_Classify_YUV420SP.py`，统计 ImageNet 验证集上的 Top-1 和 Top-5 精度：

```bash
python3 eval_Ultralytics_YOLO_Classify_YUV420SP.py \
  --model-path source/reference_bin_models/cls/yolo11n_cls_detect_bayese_640x640_nv12.bin \
  --image-path ../../../resource/datasets/ImageNet/val_images \
  --json-path yolo11n_cls_detect_bayese_640x640_nv12_py_coco2017_val_pridect.json
```

### 5. 批量评测

运行 `eval_batch.py`，对多个 `.bin` 模型执行同一类评测脚本：

```bash
python3 eval_batch.py \
  --eval-script eval_Ultralytics_YOLO_Pose_YUV420SP.py \
  --bin-paths ../model \
  --str py_coco2017_val_pridect
```

## 参数说明

### 通用参数

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--model-path` | BPU 量化模型（`.bin`）路径 | 各脚本默认不同 |
| `--image-path` | 验证图片目录路径 | 各脚本默认不同 |
| `--json-path` | 预测结果 JSON 保存路径 | 各脚本默认不同 |
| `--max-num` | 最多处理图片数量 | 各脚本默认不同 |

### 检测 / 分割 / 姿态通用参数

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--classes-num` | 类别数 | 检测/分割为 `80`，姿态为 `1` |
| `--reg` | DFL 回归通道数 | `16` |
| `--nms-thres` | NMS IoU 阈值 | `0.7` |
| `--score-thres` | 置信度阈值 | `0.25` |
| `--strides` | 特征层 stride | `8,16,32` |
| `--result-image-dump` | 是否导出可视化图片 | `False` |
| `--result-image-path` | 可视化图片保存目录 | `coco2017_image_result` |

### 分割特有参数

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--mc` | Mask 系数维度 | `32` |
| `--is-open` | 是否执行 morphology open 操作 | `False` |
| `--is-point` | 是否绘制边界点 | `True` |

### 姿态特有参数

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--nkpt` | 关键点数量 | `17` |
| `--kpt-conf-thres` | 关键点置信度阈值 | `0.5` |

### 批量脚本参数

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--eval-script` | 批量执行时使用的评测脚本 | `eval_Ultralytics_YOLO_Pose_YUV420SP.py` |
| `--bin-paths` | `.bin` 模型所在目录 | `../model` |
| `--str` | 结果名匹配关键字 | `py_coco2017_val_pridect` |

## Benchmark 结果

### 性能 Benchmark（RDK X5）

#### 目标检测

| Model | Size (Pixels) | Classes | BPU Task Latency /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | Params (M) | FLOPs (B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLOv5nu | 640x640 | 80 | 6.3 ms / 157.4 FPS (1 thread) <br> 6.8 ms / 291.8 FPS (2 threads) | 5 ms | 2.6 | 7.7 |
| YOLOv5su | 640x640 | 80 | 12.3 ms / 81.0 FPS (1 thread) <br> 18.9 ms / 105.6 FPS (2 threads) | 5 ms | 9.1 | 24.0 |
| YOLOv5mu | 640x640 | 80 | 26.5 ms / 37.7 FPS (1 thread) <br> 47.1 ms / 42.4 FPS (2 threads) | 5 ms | 25.1 | 64.2 |
| YOLOv5lu | 640x640 | 80 | 52.7 ms / 19.0 FPS (1 thread) <br> 99.1 ms / 20.1 FPS (2 threads) | 5 ms | 53.2 | 135.0 |
| YOLOv5xu | 640x640 | 80 | 91.1 ms / 11.0 FPS (1 thread) <br> 175.7 ms / 11.4 FPS (2 threads) | 5 ms | 97.2 | 246.4 |
| YOLOv8n | 640x640 | 80 | 7.0 ms / 141.9 FPS (1 thread) <br> 8.0 ms / 247.2 FPS (2 threads) | 5 ms | 3.2 | 8.7 |
| YOLOv8s | 640x640 | 80 | 13.6 ms / 73.5 FPS (1 thread) <br> 21.4 ms / 93.2 FPS (2 threads) | 5 ms | 11.2 | 28.6 |
| YOLOv8m | 640x640 | 80 | 30.6 ms / 32.6 FPS (1 thread) <br> 55.3 ms / 36.1 FPS (2 threads) | 5 ms | 25.9 | 78.9 |
| YOLOv8l | 640x640 | 80 | 59.4 ms / 16.8 FPS (1 thread) <br> 112.7 ms / 17.7 FPS (2 threads) | 5 ms | 43.7 | 165.2 |
| YOLOv8x | 640x640 | 80 | 92.4 ms / 10.8 FPS (1 thread) <br> 178.3 ms / 11.2 FPS (2 threads) | 5 ms | 68.2 | 257.8 |
| YOLOv9t | 640x640 | 80 | 6.9 ms / 144.0 FPS (1 thread) <br> 7.9 ms / 250.6 FPS (2 threads) | 5 ms | 2.1 | 8.2 |
| YOLOv9s | 640x640 | 80 | 13.0 ms / 77.0 FPS (1 thread) <br> 20.1 ms / 98.9 FPS (2 threads) | 5 ms | 7.2 | 26.9 |
| YOLOv9m | 640x640 | 80 | 32.5 ms / 30.8 FPS (1 thread) <br> 59.0 ms / 33.8 FPS (2 threads) | 5 ms | 20.1 | 76.8 |
| YOLOv9c | 640x640 | 80 | 40.3 ms / 24.8 FPS (1 thread) <br> 74.6 ms / 26.7 FPS (2 threads) | 5 ms | 25.3 | 102.7 |
| YOLOv9e | 640x640 | 80 | 119.5 ms / 8.4 FPS (1 thread) <br> 232.5 ms / 8.6 FPS (2 threads) | 5 ms | 57.4 | 189.5 |
| YOLOv10n | 640x640 | 80 | 8.7 ms / 114.2 FPS (1 thread) <br> 11.6 ms / 171.9 FPS (2 threads) | 5 ms | 2.3 | 6.7 |
| YOLOv10s | 640x640 | 80 | 14.9 ms / 67.1 FPS (1 thread) <br> 23.8 ms / 83.7 FPS (2 threads) | 5 ms | 7.2 | 21.6 |
| YOLOv10m | 640x640 | 80 | 29.4 ms / 34.0 FPS (1 thread) <br> 52.6 ms / 37.9 FPS (2 threads) | 5 ms | 15.4 | 59.1 |
| YOLOv10b | 640x640 | 80 | 40.0 ms / 25.0 FPS (1 thread) <br> 74.2 ms / 26.9 FPS (2 threads) | 5 ms | 19.1 | 92.0 |
| YOLOv10l | 640x640 | 80 | 49.8 ms / 20.1 FPS (1 thread) <br> 93.6 ms / 21.3 FPS (2 threads) | 5 ms | 24.4 | 120.3 |
| YOLOv10x | 640x640 | 80 | 68.9 ms / 14.5 FPS (1 thread) <br> 131.5 ms / 15.2 FPS (2 threads) | 5 ms | 29.5 | 160.4 |
| YOLO11n | 640x640 | 80 | 8.2 ms / 121.6 FPS (1 thread) <br> 10.5 ms / 188.9 FPS (2 threads) | 5 ms | 2.6 | 6.5 |
| YOLO11s | 640x640 | 80 | 15.7 ms / 63.4 FPS (1 thread) <br> 25.6 ms / 77.7 FPS (2 threads) | 5 ms | 9.4 | 21.5 |
| YOLO11m | 640x640 | 80 | 34.5 ms / 29.0 FPS (1 thread) <br> 63.0 ms / 31.7 FPS (2 threads) | 5 ms | 20.1 | 68.0 |
| YOLO11l | 640x640 | 80 | 45.0 ms / 22.2 FPS (1 thread) <br> 84.0 ms / 23.7 FPS (2 threads) | 5 ms | 25.3 | 86.9 |
| YOLO11x | 640x640 | 80 | 95.6 ms / 10.5 FPS (1 thread) <br> 184.8 ms / 10.8 FPS (2 threads) | 5 ms | 56.9 | 194.9 |
| YOLO12n | 640x640 | 80 | 39.4 ms / 25.3 FPS (1 thread) <br> 72.7 ms / 27.4 FPS (2 threads) | 5 ms | 2.6 | 6.5 |
| YOLO12s | 640x640 | 80 | 63.4 ms / 15.8 FPS (1 thread) <br> 120.6 ms / 16.5 FPS (2 threads) | 5 ms | 9.3 | 21.4 |
| YOLO12m | 640x640 | 80 | 102.3 ms / 9.8 FPS (1 thread) <br> 198.1 ms / 10.1 FPS (2 threads) | 5 ms | 20.2 | 67.5 |
| YOLO12l | 640x640 | 80 | 181.6 ms / 5.5 FPS (1 thread) <br> 356.4 ms / 5.6 FPS (2 threads) | 5 ms | 26.4 | 88.9 |
| YOLO12x | 640x640 | 80 | 311.9 ms / 3.2 FPS (1 thread) <br> 616.3 ms / 3.2 FPS (2 threads) | 5 ms | 59.1 | 199.0 |
| YOLOv13n | 640x640 | 80 | 44.6 ms / 22.4 FPS (1 thread) <br> 83.1 ms / 24.0 FPS (2 threads) | 5 ms | 2.5 | 6.4 |
| YOLOv13s | 640x640 | 80 | 63.6 ms / 15.7 FPS (1 thread) <br> 120.7 ms / 16.5 FPS (2 threads) | 5 ms | 9.0 | 20.8 |
| YOLOv13l | 640x640 | 80 | 171.6 ms / 5.8 FPS (1 thread) <br> 336.7 ms / 5.9 FPS (2 threads) | 5 ms | 27.6 | 88.4 |
| YOLOv13x | 640x640 | 80 | 308.4 ms / 3.2 FPS (1 thread) <br> 609.2 ms / 3.3 FPS (2 threads) | 5 ms | 64.0 | 199.2 |

#### 实例分割

| Model | Size (Pixels) | Classes | BPU Task Latency /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | Params (M) | FLOPs (B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLOv8n-Seg | 640x640 | 80 | 10.4 ms / 96.0 FPS (1 thread) <br> 10.9 ms / 181.9 FPS (2 threads) | 20 ms | 3.4 | 12.6 |
| YOLOv8s-Seg | 640x640 | 80 | 19.6 ms / 50.9 FPS (1 thread) <br> 29.0 ms / 68.7 FPS (2 threads) | 20 ms | 11.8 | 42.6 |
| YOLOv8m-Seg | 640x640 | 80 | 40.4 ms / 24.7 FPS (1 thread) <br> 70.4 ms / 28.3 FPS (2 threads) | 20 ms | 27.3 | 100.2 |
| YOLOv8l-Seg | 640x640 | 80 | 74.9 ms / 13.3 FPS (1 thread) <br> 139.4 ms / 14.3 FPS (2 threads) | 20 ms | 46.0 | 220.5 |
| YOLOv8x-Seg | 640x640 | 80 | 115.6 ms / 8.6 FPS (1 thread) <br> 221.1 ms / 9.0 FPS (2 threads) | 20 ms | 71.8 | 344.1 |
| YOLOv9c-Seg | 640x640 | 80 | 55.9 ms / 17.9 FPS (1 thread) <br> 101.3 ms / 19.7 FPS (2 threads) | 20 ms | 27.7 | 158.0 |
| YOLOv9e-Seg | 640x640 | 80 | 135.4 ms / 7.4 FPS (1 thread) <br> 260.0 ms / 7.7 FPS (2 threads) | 20 ms | 59.7 | 244.8 |
| YOLO11n-Seg | 640x640 | 80 | 11.7 ms / 85.6 FPS (1 thread) <br> 13.0 ms / 152.6 FPS (2 threads) | 20 ms | 2.9 | 10.4 |
| YOLO11s-Seg | 640x640 | 80 | 21.7 ms / 46.0 FPS (1 thread) <br> 33.1 ms / 60.3 FPS (2 threads) | 20 ms | 10.1 | 35.5 |
| YOLO11m-Seg | 640x640 | 80 | 50.3 ms / 19.9 FPS (1 thread) <br> 90.2 ms / 22.1 FPS (2 threads) | 20 ms | 22.4 | 123.3 |
| YOLO11l-Seg | 640x640 | 80 | 60.6 ms / 16.5 FPS (1 thread) <br> 110.8 ms / 18.0 FPS (2 threads) | 20 ms | 27.6 | 142.2 |
| YOLO11x-Seg | 640x640 | 80 | 129.1 ms / 7.7 FPS (1 thread) <br> 247.4 ms / 8.1 FPS (2 threads) | 20 ms | 62.1 | 319.0 |

#### 姿态估计

| Model | Size (Pixels) | Classes | BPU Task Latency /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | Params (M) | FLOPs (B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLOv8n-Pose | 640x640 | 1 | 7.0 ms / 143.1 FPS (1 thread) <br> 8.2 ms / 241.8 FPS (2 threads) | 10 ms | 3.3 | 9.2 |
| YOLOv8s-Pose | 640x640 | 1 | 14.1 ms / 70.6 FPS (1 thread) <br> 22.6 ms / 88.2 FPS (2 threads) | 10 ms | 11.6 | 30.2 |
| YOLOv8m-Pose | 640x640 | 1 | 31.5 ms / 31.7 FPS (1 thread) <br> 57.2 ms / 34.9 FPS (2 threads) | 10 ms | 26.4 | 81.0 |
| YOLOv8l-Pose | 640x640 | 1 | 60.2 ms / 16.6 FPS (1 thread) <br> 114.4 ms / 17.4 FPS (2 threads) | 10 ms | 44.4 | 168.6 |
| YOLOv8x-Pose | 640x640 | 1 | 93.9 ms / 10.7 FPS (1 thread) <br> 181.5 ms / 11.0 FPS (2 threads) | 10 ms | 69.4 | 263.2 |
| YOLO11n-Pose | 640x640 | 1 | 8.3 ms / 119.8 FPS (1 thread) <br> 10.9 ms / 182.2 FPS (2 threads) | 10 ms | 2.9 | 7.6 |
| YOLO11s-Pose | 640x640 | 1 | 16.3 ms / 61.1 FPS (1 thread) <br> 27.0 ms / 73.9 FPS (2 threads) | 10 ms | 9.9 | 23.2 |
| YOLO11m-Pose | 640x640 | 1 | 35.6 ms / 28.0 FPS (1 thread) <br> 65.4 ms / 30.5 FPS (2 threads) | 10 ms | 20.9 | 71.7 |
| YOLO11l-Pose | 640x640 | 1 | 46.3 ms / 21.6 FPS (1 thread) <br> 86.6 ms / 23.0 FPS (2 threads) | 10 ms | 26.2 | 90.7 |
| YOLO11x-Pose | 640x640 | 1 | 97.8 ms / 10.2 FPS (1 thread) <br> 189.4 ms / 10.5 FPS (2 threads) | 10 ms | 58.8 | 203.3 |

#### 图像分类

| Model | Size (Pixels) | Classes | BPU Task Latency /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | Params (M) | FLOPs (B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLOv8n-CLS | 224x224 | 1000 | 0.7 ms / 1374.6 FPS (1 thread) <br> 1.0 ms / 2023.2 FPS (2 threads) | 0.5 ms | 2.7 | 4.3 |
| YOLOv8s-CLS | 224x224 | 1000 | 1.4 ms / 701.0 FPS (1 thread) <br> 2.3 ms / 848.0 FPS (2 threads) | 0.5 ms | 6.4 | 13.5 |
| YOLOv8m-CLS | 224x224 | 1000 | 3.7 ms / 269.5 FPS (1 thread) <br> 6.9 ms / 290.6 FPS (2 threads) | 0.5 ms | 17.0 | 42.7 |
| YOLOv8l-CLS | 224x224 | 1000 | 7.9 ms / 126.6 FPS (1 thread) <br> 15.2 ms / 130.8 FPS (2 threads) | 0.5 ms | 37.5 | 99.7 |
| YOLOv8x-CLS | 224x224 | 1000 | 13.1 ms / 76.4 FPS (1 thread) <br> 25.5 ms / 78.3 FPS (2 threads) | 0.5 ms | 57.4 | 154.8 |
| YOLO11n-CLS | 224x224 | 1000 | 1.0 ms / 949.5 FPS (1 thread) <br> 1.6 ms / 1238.4 FPS (2 threads) | 0.5 ms | 2.8 | 4.2 |
| YOLO11s-CLS | 224x224 | 1000 | 2.1 ms / 484.3 FPS (1 thread) <br> 3.5 ms / 572.2 FPS (2 threads) | 0.5 ms | 6.7 | 13.0 |
| YOLO11m-CLS | 224x224 | 1000 | 3.8 ms / 262.6 FPS (1 thread) <br> 7.1 ms / 282.2 FPS (2 threads) | 0.5 ms | 11.6 | 40.3 |
| YOLO11l-CLS | 224x224 | 1000 | 5.0 ms / 200.3 FPS (1 thread) <br> 9.4 ms / 211.2 FPS (2 threads) | 0.5 ms | 14.1 | 50.4 |
| YOLO11x-CLS | 224x224 | 1000 | 10.0 ms / 100.2 FPS (1 thread) <br> 19.3 ms / 103.2 FPS (2 threads) | 0.5 ms | 29.6 | 111.3 |

### 精度 Benchmark

#### Detection (COCO2017)

| Model | PyTorch AP | Python AP |
| :--- | :--- | :--- |
| YOLOv5nu | 0.275 | 0.260 (94.55%) |
| YOLOv5su | 0.362 | 0.354 (97.79%) |
| YOLOv5mu | 0.417 | 0.407 (97.60%) |
| YOLOv5lu | 0.449 | 0.442 (98.44%) |
| YOLOv5xu | 0.458 | 0.443 (96.72%) |
| YOLOv8n | 0.306 | 0.292 (95.42%) |
| YOLOv8s | 0.384 | 0.372 (96.88%) |
| YOLOv8m | 0.433 | 0.423 (97.69%) |
| YOLOv8l | 0.454 | 0.440 (96.92%) |
| YOLOv8x | 0.465 | 0.448 (96.34%) |
| YOLOv9t | 0.357 | 0.346 (96.92%) |
| YOLOv9s | 0.460 | 0.446 (96.96%) |
| YOLOv9m | 0.504 | 0.485 (96.23%) |
| YOLOv9c | 0.530 | 0.515 (97.17%) |
| YOLOv9e | 0.555 | 0.530 (95.50%) |
| YOLOv10n | 0.387 | 0.357 (92.25%) |
| YOLOv10s | 0.469 | 0.444 (94.67%) |
| YOLOv10m | 0.510 | 0.482 (94.50%) |
| YOLOv10b | 0.525 | 0.504 (96.00%) |
| YOLOv10l | 0.540 | 0.517 (95.74%) |
| YOLOv10x | 0.541 | 0.522 (96.49%) |
| YOLO11n | 0.323 | 0.308 (95.36%) |
| YOLO11s | 0.394 | 0.380 (96.45%) |
| YOLO11m | 0.437 | 0.422 (96.57%) |
| YOLO11l | 0.452 | 0.432 (95.58%) |
| YOLO11x | 0.466 | 0.446 (95.71%) |
| YOLO12n | 0.410 | 0.383 (93.41%) |
| YOLO12s | 0.487 | 0.465 (95.48%) |
| YOLO12m | 0.533 | 0.513 (96.25%) |
| YOLO12l | 0.545 | 0.523 (95.96%) |
| YOLO12x | 0.557 | 0.532 (95.51%) |
| YOLOv13n | 0.409 | 0.385 (94.13%) |
| YOLOv13s | 0.485 | 0.458 (94.43%) |
| YOLOv13l | 0.538 | 0.510 (94.80%) |
| YOLOv13x | 0.551 | 0.526 (95.46%) |

### Segmentation (COCO2017)

| Model | PyTorch Box / Mask | Python Box / Mask |
| :--- | :--- | :--- |
| YOLOv8n-Seg | 0.300 / 0.241 | 0.284 / 0.219 |
| YOLOv8s-Seg | 0.380 / 0.299 | 0.371 / 0.287 |
| YOLOv8m-Seg | 0.423 / 0.330 | 0.408 / 0.311 |
| YOLOv8l-Seg | 0.444 / 0.344 | 0.431 / 0.332 |
| YOLOv8x-Seg | 0.456 / 0.351 | 0.439 / 0.336 |
| YOLOv9c-Seg | 0.446 / 0.345 | 0.423 / 0.321 |
| YOLOv9e-Seg | 0.471 / 0.118 | 0.332 / 0.268 |
| YOLO11n-Seg | 0.319 / 0.258 | 0.296 / 0.227 |
| YOLO11s-Seg | 0.388 / 0.306 | 0.377 / 0.291 |
| YOLO11m-Seg | 0.436 / 0.340 | 0.422 / 0.322 |
| YOLO11l-Seg | 0.452 / 0.350 | 0.432 / 0.328 |
| YOLO11x-Seg | 0.466 / 0.358 | 0.447 / 0.338 |

### Pose Estimation (COCO2017)

| Model | PyTorch AP | Python AP |
| :--- | :--- | :--- |
| YOLOv8n-Pose | 0.476 | 0.462 |
| YOLOv8s-Pose | 0.578 | 0.553 |
| YOLOv8m-Pose | 0.631 | 0.605 |
| YOLOv8l-Pose | 0.656 | 0.636 |
| YOLOv8x-Pose | 0.670 | 0.655 |
| YOLO11n-Pose | 0.465 | 0.452 |
| YOLO11s-Pose | 0.560 | 0.530 |
| YOLO11m-Pose | 0.626 | 0.600 |
| YOLO11l-Pose | 0.636 | 0.619 |
| YOLO11x-Pose | 0.672 | 0.654 |

### Classification (ImageNet2012)

| Model | PyTorch TOP1 / TOP5 | Python TOP1 / TOP5 |
| :--- | :--- | :--- |
| YOLOv8n-CLS | 0.690 / 0.883 | 0.525 / 0.762 |
| YOLOv8s-CLS | 0.738 / 0.917 | 0.611 / 0.837 |
| YOLOv8m-CLS | 0.768 / 0.935 | 0.682 / 0.883 |
| YOLOv8l-CLS | 0.768 / 0.935 | 0.724 / 0.909 |
| YOLOv8x-CLS | 0.790 / 0.946 | 0.737 / 0.917 |
| YOLO11n-CLS | 0.700 / 0.894 | 0.495 / 0.736 |
| YOLO11s-CLS | 0.754 / 0.927 | 0.665 / 0.873 |
| YOLO11m-CLS | 0.773 / 0.939 | 0.695 / 0.896 |
| YOLO11l-CLS | 0.783 / 0.943 | 0.707 / 0.902 |
| YOLO11x-CLS | 0.795 / 0.949 | 0.732 / 0.917 |

## 性能测试说明

1. 本文中的性能数据均基于 YUV420SP（NV12）输入模型测试得到。
2. BPU latency 和 BPU throughput 均在板端测试。
   - 单线程延迟表示单线程、单 BPU 核心处理单帧的理想延迟。
   - 多线程吞吐表示多个线程并发向 BPU 提交任务时的 FPS。
   - 表中的吞吐数据通常取自线程数继续增加时 FPS 提升已经不明显的点。
3. 性能测试命令示例：
   ```bash
   hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin
   python3 ../../../resource/tools/batch_perf/batch_perf.py --max 3 --file source/reference_hbm_models/
   ```
4. 测试时建议将开发板设置到高性能状态，包括 CPU governor 和 BPU governor 的性能模式。

## 精度测试说明

1. 所有精度数据均基于官方未修改版 `pycocotools` 计算得到。
2. 检测与分割任务使用的指标为 `Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ]`，姿态任务使用的指标为 `Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=20 ]`。
3. 检测、分割、姿态默认使用 COCO2017 验证集 5000 张图片；分类默认使用 ImageNet 验证集。
4. 推理在板端执行，预测结果会导出为 JSON 文件，再使用离线工具进行评测。
5. 默认评测阈值为 `score_thres=0.25`、`nms_thres=0.7`。
6. `pycocotools` 计算出的精度通常会略低于 `ultralytics` 官方工具，这是因为两者的 AP 积分方式不同。
7. 部署模型使用 YUV420SP（NV12）输入，相比原始 RGB 浮点链路可能存在轻微精度变化。
8. 本文档中的 benchmark 数据基于 PTQ（Post-Training Quantization）和 50 张校准图片得到，用于常规部署验证，不代表通过进一步调优或 QAT 能达到的精度上限。

## 验证结果汇总

本 sample 文档中列出的模型均已在 RDK X5 板端完成验证：

- Detect: 35 个模型
- Seg: 12 个模型
- Pose: 10 个模型
- Classification: 10 个模型

合计：`67 / 67` 模型通过板端验证。
