[English](./README.md) | 简体中文

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

## 使用方法 (Usage)

本目录提供了针对不同任务的评估脚本。运行脚本前，请确保已按照上文准备好环境和数据集。

### 1. 目标检测 (Object Detection)

使用 `eval_yolo26_det.py` 脚本评估检测模型在 COCO 数据集上的 mAP。

```bash
python3 eval_yolo26_det.py \
  --model-path ../model/nash-e/yolo26n_detect_nashe_640x640_nv12.hbm \
  --image-dir ../../../../datasets/coco/val2017 \
  --annotation ../../../../datasets/coco/annotations/instances_val2017.json \
  --conf-thres 0.25 \
  --nms-thres 0.7
```

**参数说明**:
- `--model-path`: 量化后的模型文件路径 (.hbm)。
- `--image-dir`: 验证集图片目录。
- `--annotation`: COCO 格式的标注文件路径。
- `--conf-thres`: 置信度阈值 (默认 0.25, 精度测试建议设低以获取更全的 recall)。
- `--nms-thres`: NMS 阈值。

### 2. 图像分类 (Image Classification)

使用 `eval_yolo26_cls.py` 脚本评估分类模型在 ImageNet 数据集上的 Top-1/Top-5 准确率。

```bash
python3 eval_yolo26_cls.py \
  --model-path ../model/nash-e/yolo26n_cls_nashe_224x224_nv12.hbm \
  --image-dir ../../../../datasets/imagenet/val \
  --val-txt ../../../../datasets/imagenet/val.txt \
  --topk 5
```

**参数说明**:
- `--val-txt`: 包含文件名和标签映射的文本文件 (格式: `filename label_id`)。

### 3. 实例分割 (Instance Segmentation)

使用 `eval_yolo26_seg.py` 脚本评估分割模型。

```bash
python3 eval_yolo26_seg.py \
  --model-path ../model/nash-e/yolo26n_seg_nashe_640x640_nv12.hbm \
  --image-dir ../../../../datasets/coco/val2017 \
  --annotation ../../../../datasets/coco/annotations/instances_val2017.json
```

### 4. 姿态估计 (Pose Estimation)

使用 `eval_yolo26_pose.py` 脚本评估人体关键点模型。

```bash
python3 eval_yolo26_pose.py \
  --model-path ../model/nash-e/yolo26n_pose_nashe_640x640_nv12.hbm \
  --image-dir ../../../../datasets/coco/val2017 \
  --annotation ../../../../datasets/coco/annotations/person_keypoints_val2017.json
```

### 5. 旋转框检测 (OBB)

使用 `eval_yolo26_obb.py` 脚本评估旋转目标检测模型 (通常使用 DOTA 数据集格式)。

```bash
python3 eval_yolo26_obb.py \
  --model-path ../model/nash-e/yolo26n_obb_nashe_640x640_nv12.hbm \
  --image-dir ../../../../datasets/dotav1/val \
```

## 基准测试结果 (Benchmark Results)

### RDK S100/S100P 性能数据 (Performance @ NV12)

| 设备 | 模型 | 尺寸 <br> (Pixels) | 类别数 | BPU 任务延迟 / <br> BPU 吞吐量 (线程) | CPU 延迟 | 参数量 <br> (M) | 计算量 <br> (G) |
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

### RDK S100 精度数据 (Accuracy @ NV12 - Detection)

| Device | Model | Accuracy bbox-all <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-small <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-medium <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-large <br> mAP @.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S100 | YOLO26n Detect | 0.319 / 0.286 (89.7 %) | 0.107 / 0.083 (77.6 %) | 0.349 / 0.304 (87.1 %) | 0.508 / 0.473 (93.1 %) |
| S100 | YOLO26s Detect | 0.395 / 0.362 (91.6 %) | 0.183 / 0.163 (89.1 %) | 0.440 / 0.402 (91.4 %) | 0.583 / 0.524 (89.9 %) |
| S100 | YOLO26m Detect | 0.442 / 0.413 (93.4 %) | 0.242 / 0.202 (83.5 %) | 0.489 / 0.456 (93.3 %) | 0.629 / 0.603 (95.9 %) |
| S100 | YOLO26l Detect | 0.456 / 0.440 (96.5 %) | 0.260 / 0.230 (88.5 %) | 0.499 / 0.489 (98.0 %) | 0.627 / 0.623 (99.4 %) |
| S100 | YOLO26x Detect | 0.484 / 0.449 (92.8 %) | 0.292 / 0.246 (84.2 %) | 0.528 / 0.488 (92.4 %) | 0.669 / 0.646 (96.6 %) |

### RDK S100P 精度数据 (Accuracy @ RGB - Detection)

| Device | Model | Accuracy bbox-all <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-small <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-medium <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy bbox-large <br> mAP @.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S100P | YOLO26n Detect | 0.319 / 0.290 (91.0 %) | 0.107 / 0.087 (81.7 %) | 0.349 / 0.313 (89.8 %) | 0.508 / 0.463 (91.1 %) |
| S100P | YOLO26s Detect | 0.395 / 0.367 (93.0 %) | 0.183 / 0.174 (94.8 %) | 0.440 / 0.410 (93.1 %) | 0.583 / 0.530 (91.0 %) |
| S100P | YOLO26m Detect | 0.442 / 0.421 (95.2 %) | 0.242 / 0.224 (92.6 %) | 0.489 / 0.460 (94.0 %) | 0.629 / 0.603 (95.8 %) |
| S100P | YOLO26l Detect | 0.456 / 0.437 (96.0 %) | 0.260 / 0.234 (89.8 %) | 0.499 / 0.484 (97.1 %) | 0.627 / 0.609 (97.0 %) |
| S100P | YOLO26x Detect | 0.484 / 0.466 (96.2 %) | 0.292 / 0.271 (92.8 %) | 0.528 / 0.502 (95.1 %) | 0.669 / 0.654 (97.8 %) |

### RDK S100 精度数据 (Accuracy @ NV12 - Pose Estimation)

| Device | Model | Accuracy kpt-all <br> mAP @.50:.95 <br> (BPU Python) | Accuracy kpt-medium <br> mAP @.50:.95 <br> (BPU Python) | Accuracy kpt-large <br> mAP @.50:.95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- | :--- |
| S100 | YOLO26n Pose | 0.504 | 0.412 | 0.647 |
| S100 | YOLO26s Pose | 0.575 | 0.498 | 0.697 |
| S100 | YOLO26m Pose | 0.620 | 0.554 | 0.737 |
| S100 | YOLO26l Pose | 0.646 | 0.579 | 0.744 |
| S100 | YOLO26x Pose | 0.663 | 0.601 | 0.775 |

### RDK S100P 精度数据 (Accuracy @ NV12 - Pose Estimation)

| Device | Model | Accuracy kpt-all <br> mAP @.50:.95 <br> (BPU Python) | Accuracy kpt-medium <br> mAP @.50:.95 <br> (BPU Python) | Accuracy kpt-large <br> mAP @.50:.95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- | :--- |
| S100P | YOLO26n Pose | - | - | - |

### RDK S100 精度数据 (Accuracy @ NV12 - Segmentation)

| Device | Model | Accuracy mask-all <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-small <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-medium <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-large <br> mAP @.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S100 | YOLO26n Seg | - / 0.254 | - / 0.057 | - / 0.269 | - / 0.434 |
| S100 | YOLO26s Seg | - / 0.330 | - / 0.119 | - / 0.367 | - / 0.510 |
| S100 | YOLO26m Seg | - / 0.356 | - / 0.148 | - / 0.399 | - / 0.536 |
| S100 | YOLO26l Seg | - / 0.375 | - / 0.164 | - / 0.419 | - / 0.560 |
| S100 | YOLO26x Seg | - / 0.381 | - / 0.176 | - / 0.426 | - / 0.576 |

### RDK S100P 精度数据 (Accuracy @ NV12 - Segmentation)

| Device | Model | Accuracy mask-all <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-small <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-medium <br> mAP @.50:.95 <br> (FP32 / BPU Python) | Accuracy mask-large <br> mAP @.50:.95 <br> (FP32 / BPU Python) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S100P | YOLO26n Seg | - / - (- %) | - / - (- %) | - / - (- %) | - / - (- %) |

### RDK S100 精度数据 (Accuracy @ NV12 - Classification)

| Device | Model | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :--- | :--- | :--- |
| S100 | YOLO26n Cls | 0.6165 | 0.8359 |
| S100 | YOLO26s Cls | 0.6854 | 0.8853 |
| S100 | YOLO26m Cls | 0.7194 | 0.9080 |
| S100 | YOLO26l Cls | 0.7369 | 0.9168 |
| S100 | YOLO26x Cls | 0.7432 | 0.9222 |

### RDK S100P 精度数据 (Accuracy @ NV12 - Classification)

| Device | Model | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :--- | :--- | :--- |
| S100P | YOLO26n Cls | - | - |

### RDK S100 精度数据 (Accuracy @ NV12 - OBB)

| Device | Model | Accuracy mAP50 <br> (BPU Python) | Accuracy mAP50-95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- |
| S100 | YOLO26n Obb | - | - |
| S100 | YOLO26s Obb | - | - |
| S100 | YOLO26m Obb | - | - |
| S100 | YOLO26l Obb | - | - |
| S100 | YOLO26x Obb | - | - |

### RDK S100P 精度数据 (Accuracy @ NV12 - OBB)

| Device | Model | Accuracy mAP50 <br> (BPU Python) | Accuracy mAP50-95 <br> (BPU Python) |
| :--- | :--- | :--- | :--- |
| S100P | YOLO26n Obb | - | - |


## 输出指标

- **检测/分割/姿态**: 输出 AP @ IoU=0.50:0.95 (all, small, medium, large), AP @ 0.5, AP @ 0.75 以及 Recall 指标。
- **分类**: 输出 Top-1 Accuracy, Top-5 Accuracy 以及推理 FPS。
- **OBB**: 输出 mAP50 和 mAP50-95 (DOTA metric)。

## 性能测试说明 (Performance Test Instructions)

- **设备 (Device)**: 测试所用的硬件设备，如 S100、S100P 等。
- **模型 (Model)**: 测试所用的模型，如 YOLO26n、YOLO26s 等。
- **类别数 (Classes)**: 模型检测的类别数量，与 COCO2017 或 ImageNet-1k 数据集保持一致。
- **BPU 任务延迟 / 吞吐量 (BPU Task Latency / Throughput)**:
  - **单线程延迟**: 单帧图像在单线程、单 BPU 核心下的理想推理延迟。
  - **多线程吞吐量**: 多线程同时向 BPU 提交任务时达到的每秒处理帧数 (FPS)。通常 2 线程能较好地平衡低延迟和高 BPU 利用率。
  - **测试命令**: 使用开发工具包中的 `hrt_model_exec` 工具：`hrt_model_exec perf --thread_num 2 --model_file <model.hbm>`。该工具测量从任务提交到完成的时间，并考虑了缓存预热。
- **CPU 延迟 (单核)**: 后处理耗时。与检测到的目标数量正相关（本表数据基于目标数 < 100 的场景）。Python 和 C++ 实现均经过深度优化。
- **内存管理**: 在流式推理中，输入/输出内存应分配一次并循环复用。延迟测量中不应包含内存分配和释放的时间。
- **参数量 (M) & 计算量 (G)**: 原始 FP32 模型的参数量和计算量（源自 Ultralytics YOLO 导出日志）。由于编译器优化，BPU 实际计算量可能有所不同。

## 精度测试说明 (Accuracy Test Instructions)

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
