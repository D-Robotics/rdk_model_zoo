[English](./README.md) | 简体中文

# 模型文件

本目录保存 Ultralytics YOLO 在 RDK X5 平台上使用的预编译 `.bin` 模型和下载脚本。

## 目录结构

```text
.
|-- download_model.sh      # 下载默认模型
|-- fulldownload.sh        # 下载全部支持模型
|-- README.md              # 英文文档
`-- README_cn.md           # 中文文档
```

## 下载模型

下载 `runtime/python/run.sh` 默认使用的模型：

```bash
chmod +x download_model.sh
./download_model.sh
```

下载本 sample 支持的全部模型：

```bash
chmod +x fulldownload.sh
./fulldownload.sh
```

## 说明

- `RDK X5` 的推理模型格式为 `.bin`
- 所有模型均使用 packed `NV12` 输入
- 当前模型下载地址为：
  `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ultralytics_YOLO/`

## 默认模型

- `detect`: `yolo11n_detect_bayese_640x640_nv12.bin`
- `seg`: `yolo11n_seg_bayese_640x640_nv12.bin`
- `pose`: `yolo11n_pose_bayese_640x640_nv12.bin`
- `cls`: `yolo11n_cls_detect_bayese_640x640_nv12.bin`

## 模型列表

### Detection

- `yolov5{n/s/m/l/x}u_detect_bayese_640x640_nv12.bin`
- `yolov8{n/s/m/l/x}_detect_bayese_640x640_nv12.bin`
- `yolov9{t/s/m/c/e}_detect_bayese_640x640_nv12.bin`
- `yolov10{n/s/m/b/l/x}_detect_bayese_640x640_nv12.bin`
- `yolo11{n/s/m/l/x}_detect_bayese_640x640_nv12.bin`
- `yolo12{n/s/m/l/x}_detect_bayese_640x640_nv12.bin`
- `yolov13{n/s/l/x}_detect_bayese_640x640_nv12.bin`

### Instance Segmentation

- `yolov8{n/s/m/l/x}_seg_bayese_640x640_nv12.bin`
- `yolov9{c/e}_seg_bayese_640x640_nv12.bin`
- `yolo11{n/s/m/l/x}_seg_bayese_640x640_nv12.bin`

### Pose Estimation

- `yolov8{n/s/m/l/x}_pose_bayese_640x640_nv12.bin`
- `yolo11{n/s/m/l/x}_pose_bayese_640x640_nv12.bin`

### Classification

- `yolov8{n/s/m/l/x}_cls_detect_bayese_640x640_nv12.bin`
- `yolo11{n/s/m/l/x}_cls_detect_bayese_640x640_nv12.bin`
