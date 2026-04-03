# 模型文件说明

本目录包含 RDK X5 平台使用的 YOLO26 预转换模型文件说明和下载脚本。

## 目录结构

```text
.
├── download_model.sh      # 下载 nano 模型
├── fulldownload.sh        # 下载全部模型
└── README.md              # 文档说明
```

## 下载模型

下载 RDK X5 预编译 YOLO26 模型：

```bash
chmod +x download_model.sh
./download_model.sh
```

下载全部 `n / s / m / l / x` 模型：

```bash
chmod +x fulldownload.sh
./fulldownload.sh
```

## 说明

- RDK X5 推理模型格式为 `.bin`
- 所有模型输入格式均为 `NV12`
- 当前模型下载地址为：
  `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/Ultralytics_YOLO_OE_1.2.8/`

## 模型列表

### 目标检测
- `yolo26{n/s/m/l/x}_detect_bayese_640x640_nv12.bin`

### 实例分割
- `yolo26{n/s/m/l/x}_seg_bayese_640x640_nv12.bin`

### 姿态估计
- `yolo26{n/s/m/l/x}_pose_bayese_640x640_nv12.bin`

### 旋转框检测
- `yolo26{n/s/m/l/x}_obb_bayese_640x640_nv12.bin`

### 图像分类
- `yolo26{n/s/m/l/x}_cls_bayese_224x224_nv12.bin`
