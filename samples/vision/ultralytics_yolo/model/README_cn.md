[English](./README.md) | 简体中文

# 模型文件

本目录包含 RDK S100/S100P 的预编译 YOLO 模型文件和下载脚本。

## 目录结构

```text
.
├── download_model.sh      # 下载指定模型
└── README.md              # 文档说明
```

## 下载模型

通过指定 SoC、YOLO 变体、任务和可选模型尺寸来下载特定模型：

```bash
bash download_model.sh [soc] [yolo_type] [task] [model_size]

# 示例：
bash download_model.sh s100 yolo11 detect
bash download_model.sh s100p yolov8 seg
bash download_model.sh s100 yolov8 pose
bash download_model.sh s100p yolo11 cls
bash download_model.sh s100p yolov5u detect x
```

## 说明

- RDK S100/S100P 推理模型格式为 `.hbm`。
- 所有模型使用 NV12 输入（Y + UV 两个独立输入张量）。
- 模型后缀因平台不同而异：S100 为 `nashe`，S100P 为 `nashm`。
- 主下载 URL 基础路径：`https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.7.0/{nash-e|nash-m}/`
- `model_size` 支持 `n/s/m/l/x`；YOLOv10 额外支持 `b`。

## 模型列表

### 目标检测

- `yolov5{n|s|m|l|x}u_detect_{nashe|nashm}_640x640_nv12.hbm`
- `yolov8{n|s|m|l|x}_detect_{nashe|nashm}_640x640_nv12.hbm`
- `yolov10{n|s|m|b|l|x}_detect_{nashe|nashm}_640x640_nv12.hbm`
- `yolo11{n|s|m|l|x}_detect_{nashe|nashm}_640x640_nv12.hbm`
- `yolo12{n|s|m|l|x}_detect_{nashe|nashm}_640x640_nv12.hbm`

### 实例分割

- `yolov8{n|s|m|l|x}_seg_{nashe|nashm}_640x640_nv12.hbm`
- `yolo11{n|s|m|l|x}_seg_{nashe|nashm}_640x640_nv12.hbm`

### 姿态估计

- `yolov8{n|s|m|l|x}_pose_{nashe|nashm}_640x640_nv12.hbm`
- `yolo11{n|s|m|l|x}_pose_{nashe|nashm}_640x640_nv12.hbm`

### 图像分类

- `yolov8{n|s|m|l|x}_cls_{nashe|nashm}_640x640_nv12.hbm`
- `yolo11{n|s|m|l|x}_cls_{nashe|nashm}_640x640_nv12.hbm`
