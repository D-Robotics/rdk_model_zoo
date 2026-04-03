# YOLO26 模型评估 (Evaluation)

本目录包含用于评估 YOLO26 各类任务模型精度的脚本，支持在 RDK 硬件上直接运行并输出标准指标。

## 环境准备

- **Python 环境**: 确保 RDK 已经安装了 Python 3。
- **依赖库**:
  - `pycocotools`: 用于 COCO 数据集 (检测、分割、姿态) 的 mAP 计算。
    ```bash
    pip install pycocotools
    ```
  - `opencv-python`, `numpy` 等基础库。

## 数据集准备

脚本默认从 `datasets/` 目录读取数据，请确保数据集路径正确：
- **检测/分割/姿态**: [COCO val2017](../../../../datasets/coco/README.md)
- **分类**: [ImageNet val](../../../../datasets/imagenet/README.md)

## 使用方法

### 1. 目标检测评估 (Detection)
运行 `eval_yolo26_det.py`，计算 COCO mAP：
```bash
python3 eval_yolo26_det.py --model-path ../model/yolo26n_det.bin --limit 100
```

### 2. 实例分割评估 (Segmentation)
运行 `eval_yolo26_seg.py`，计算 Seg mAP：
```bash
python3 eval_yolo26_seg.py --model-path ../model/yolo26n_seg.bin --limit 100
```

### 3. 姿态估计评估 (Pose)
运行 `eval_yolo26_pose.py`，计算 Keypoints mAP：
```bash
python3 eval_yolo26_pose.py --model-path ../model/yolo26n_pose.bin --limit 100
```

### 4. 图像分类评估 (Classification)
运行 `eval_yolo26_cls.py`，计算 Top-1 和 Top-5 准确率：
```bash
python3 eval_yolo26_cls.py --model-path ../model/yolo26n_cls.bin --image-path /path/to/imagenet/val
```

### 5. 旋转目标检测评估 (OBB)
运行 `eval_yolo26_obb.py`，生成包含 8 点坐标多边形的预测结果 JSON（适用于 DOTA 等数据集）：
```bash
python3 eval_yolo26_obb.py --model-path ../model/yolo26n_obb.bin --limit 100
```

## 参数说明

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--model-path` | BPU 模型的路径 (.bin) | 必填 |
| `--image-path` | 验证集图片目录 | 对应任务默认路径 |
| `--ann-path` | 官方标注 JSON 文件路径 | 对应任务默认路径 |
| `--json-save-path` | 推理结果保存路径 (.json) | yolo26_xxx_results.json |
| `--conf-thres` | 置信度阈值 (建议设为 0.001 以获得准确 mAP) | 0.001 |
| `--limit` | 限制处理的图片数量 (0 表示全部) | 0 |

## 输出指标

- **检测/分割/姿态**: 输出 AP @ IoU=0.50:0.95 (all, small, medium, large), AP @ 0.5, AP @ 0.75 以及 Recall 指标。
- **分类**: 输出 Top-1 Accuracy, Top-5 Accuracy 以及推理 FPS。

## 基准测试结果 (Benchmark Results)

### RDK X5 性能数据 (Performance @ NV12)

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

### RDK X5 精度数据 (Accuracy @ NV12 - Detection)

| Device | Model | Accuracy bbox-all <br> mAP@.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-small <br> mAP@.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-medium <br> mAP@.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-large <br> mAP@.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| X5 | YOLO26n Detect | 0.319 / 0.284 (89.0 %) | 0.107 / 0.075 (70.1 %) | 0.349 / 0.299 (85.7 %) | 0.508 / 0.467 (91.9 %) |
| X5 | YOLO26s Detect | 0.395 / 0.357 (90.4 %) | 0.183 / 0.154 (84.2 %) | 0.440 / 0.393 (89.3 %) | 0.583 / 0.534 (91.6 %) |
| X5 | YOLO26m Detect | 0.442 / 0.413 (93.4 %) | 0.242 / 0.206 (85.1 %) | 0.489 / 0.454 (92.8 %) | 0.629 / 0.605 (96.1 %) |
| X5 | YOLO26l Detect | 0.456 / 0.431 (94.5 %) | 0.260 / 0.215 (82.7 %) | 0.499 / 0.479 (96.0 %) | 0.627 / 0.618 (98.6 %) |
| X5 | YOLO26x Detect | 0.484 / 0.438 (90.5 %) | 0.292 / 0.230 (78.8 %) | 0.528 / 0.479 (90.7 %) | 0.669 / 0.635 (94.9 %) |

### RDK X5 精度数据 (Accuracy @ NV12 - Segmentation)

| Device | Model | Accuracy mask-all <br> mAP@.50:.95 <br> (BPU Python) | Accuracy mask-small <br> mAP@.50:.95 <br> (BPU Python) | Accuracy mask-medium <br> mAP@.50:.95 <br> (BPU Python) | Accuracy mask-large <br> mAP@.50:.95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| X5 | YOLO26n Seg | 0.285 | 0.090 | 0.307 | 0.464 |

### RDK X5 精度数据 (Accuracy @ NV12 - Pose Estimation)

| Device | Model | Accuracy kpt-all <br> mAP@.50:.95 <br> (BPU Python) | Accuracy kpt-medium <br> mAP@.50:.95 <br> (BPU Python) | Accuracy kpt-large <br> mAP@.50:.95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- | :--- |
| X5 | YOLO26n Pose | 0.498 | 0.404 | 0.647 |

## 性能测试说明 (Performance Test Instructions)

- **Device列和Model列**: 含义与 Performance Test Instructions 章节的含义相同。

- **计算工具**: 精度数据使用微软官方的无修改的 `pycocotools` 库进行计算。

- **测评模式**:

  - 目标检测 (Object Detection): `iouType="bbox"`

  - 实例分割 (Instance Segmentation): `iouType="bbox"` 和 `iouType="segm"`

  - 人体关键点估计 (Pose Estimation): `iouType="keypoints"`

- **指标含义**:

  - `Accuracy bbox-all mAP @.50:.95` 取自 `Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ]`。

  - `Accuracy bbox-small mAP @.50:.95` 取自 `Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]`。

  - `Accuracy bbox-medium mAP @.50:.95` 取自 `Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]`。

  - `Accuracy bbox-large mAP @.50:.95` 取自 `Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]`。

- **AP vs AR**: AP 更关注“质量”（既要找到目标 Recall，又要框得准且类别对 Precision）；AR 更关注“数量”（只要框住就算，不惩罚误检）。本评估统一取 **AP 指标** 来衡量模型精度。

- **测试数据**: 使用 COCO2017 验证集的 5000 张图片，在板端直接推理，dump 保存为 JSON 文件后送入 `pycocotools` 计算。分数的阈值为 0.25，NMS 的阈值为 0.7。

- **精度差异说明**:

  - `pycocotools` 计算的精度通常比 `ultralytics` 官方工具略低，这是由于 `pycocotools` 取矩形面积而 `ultralytics` 取梯形面积计算 AP 曲线下面积。我们主要关注同一套计算方式下定点模型与浮点模型的对比，以评估量化损失。

  - **分类任务**: 使用 ImageNet-1k 数据集，通过 Top-1 和 Top-5 两个指标来评估。

  - **色彩空间转化**: BPU 模型将 NCHW-RGB888 输入转换为 YUV420SP (NV12) 后会引入细微精度损失。在训练时加入色彩空间转化损失可缓解此问题。

  - **接口差异**: Python 接口和 C/C++ 接口由于在 `memcpy` 转化过程中对浮点数处理方式不同，可能存在细微精度差异。

- **量化说明**: 本表格数据基于 **PTQ (训练后量化)**，使用 50 张图片进行校准。这旨在模拟普通开发者第一次直接编译的精度情况，未进行深度精度调优或 QAT (量化感知训练)，不代表精度的理论上限。
