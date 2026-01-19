[English](./README.md) | 简体中文

# YOLO26 RDK Model Zoo 示例

本目录包含了在 RDK X5 硬件上导出、转换和运行 YOLO26 模型（目标检测、实例分割、姿态估计、图像分类）的脚本。

## 项目结构

```
.
├── cls/                # 图像分类任务脚本
│   ├── export_yolo26_cls_bpu.py
│   └── YOLO26_Cls_Inference.py
├── detect/             # 目标检测任务脚本
│   ├── export_yolo26_detect_bpu.py
│   └── YOLO26_Detect_Inference.py
├── pose/               # 姿态估计任务脚本
│   ├── export_yolo26_pose_bpu.py
│   └── YOLO26_Pose_Inference.py
├── seg/                # 实例分割任务脚本
│   ├── export_yolo26_seg_bpu.py
│   └── YOLO26_Seg_Inference.py
├── mapper.py           # 用于将 ONNX 转换为 BPU BIN 模型的通用工具
└── README_cn.md
```

## 工作流程概览

部署过程主要包含三个步骤：
1.  **导出 (Export)**：将 PyTorch (`.pt`) 模型导出为针对 BPU 优化的 ONNX 格式。
2.  **转换 (Convert)**：使用 `mapper.py` 将 ONNX 模型转换为兼容 BPU 的 `.bin` 模型。
3.  **推理 (Inference)**：在 RDK 板端使用特定任务的推理脚本运行模型。

## 1. 目标检测 (Detect)

### 第一步：导出 ONNX
```bash
cd detect
python3 export_yolo26_detect_bpu.py
# 生成文件示例: yolo26n_detect_bpu.onnx
```

### 第二步：转换为 BIN
使用根目录下的 `mapper.py` 工具。
```bash
cd ..
python3 mapper.py --onnx detect/yolo26n_detect_bpu.onnx --cal-images /path/to/calibration_data
# 生成文件示例: yolo26n_detect_bayese_640x640_nv12.bin
```

### 第三步：板端推理
将生成的 `.bin` 文件拷贝到 RDK 板端。
```bash
python3 detect/YOLO26_Detect_Inference.py --model-path yolo26n_detect_bayese_640x640_nv12.bin --test-img bus.jpg
```

---

## 2. 实例分割 (Seg)

### 第一步：导出 ONNX
```bash
cd seg
python3 export_yolo26_seg_bpu.py
```

### 第二步：转换为 BIN
```bash
cd ..
python3 mapper.py --onnx seg/yolo26n_seg_bpu.onnx --cal-images /path/to/calibration_data
```

### 第三步：板端推理
```bash
python3 seg/YOLO26_Seg_Inference.py --model-path yolo26n_seg_bayese_640x640_nv12.bin --test-img bus.jpg
```

---

## 3. 姿态估计 (Pose)

### 第一步：导出 ONNX
```bash
cd pose
python3 export_yolo26_pose_bpu.py
```

### 第二步：转换为 BIN
```bash
cd ..
python3 mapper.py --onnx pose/yolo26n_pose_bpu.onnx --cal-images /path/to/calibration_data
```

### 第三步：板端推理
```bash
python3 pose/YOLO26_Pose_Inference.py --model-path yolo26n_pose_bayese_640x640_nv12.bin --test-img bus.jpg
```

---

## 4. 图像分类 (Cls)

### 第一步：导出 ONNX
```bash
cd cls
python3 export_yolo26_cls_bpu.py
```

### 第二步：转换为 BIN
```bash
cd ..
python3 mapper.py --onnx cls/yolo26n_cls_bpu.onnx --cal-images /path/to/calibration_data
```

### 第三步：板端推理
```bash
python3 cls/YOLO26_Cls_Inference.py --model-path yolo26n_cls_bayese_224x224_nv12.bin --test-img bus.jpg
```

## 关于 `mapper.py`
`mapper.py` 脚本是 `hb_mapper` 的封装工具，负责处理校准数据准备和模型转换。
- `--onnx`：输入 ONNX 模型的路径。
- `--cal-images`：包含校准图像（JPG/PNG）的目录路径。
- `--output-dir`：（可选）保存输出 `.bin` 模型的目录。
- `--quantized`：（可选）`int8`（默认）或 `int16`。

请确保在安装了 `hb_mapper` 的环境（例如 OpenExplore Docker）中运行 `mapper.py`。
