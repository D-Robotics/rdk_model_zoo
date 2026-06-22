[English](./README.md) | 简体中文

# 模型文件

本目录包含 RDK S100/S100P/S600 的预编译 YOLO26 模型文件和下载脚本。

## 目录结构

```text
.
├── download_model.sh      # 下载预编译模型
└── README.md              # 文档说明
```

## 下载模型

运行以下命令下载 RDK S100/S100P/S600 的预编译 YOLO26 模型：

```bash
bash download_model.sh
```

脚本会自动识别板卡，并将模型下载到 `./nash-e/`、`./nash-m/` 或 `./nash-p/` 目录。

## 说明

- RDK S100/S100P/S600 推理模型格式为 `.hbm`。
- 所有模型使用 NV12 输入（Y + UV 两个独立输入张量）。
- 模型后缀因平台不同而异：S100 为 `nashe`，S100P 为 `nashm`，S600 为 `nashp`。

## 模型列表

### 目标检测
- `yolo26n_detect_{nashe|nashm|nashp}_640x640_nv12.hbm`

### 实例分割
- `yolo26n_seg_{nashe|nashm|nashp}_640x640_nv12.hbm`

### 姿态估计
- `yolo26n_pose_{nashe|nashm|nashp}_640x640_nv12.hbm`

### 旋转框检测
- `yolo26n_obb_{nashe|nashm|nashp}_640x640_nv12.hbm`

### 图像分类
- `yolo26n_cls_{nashe|nashm|nashp}_224x224_nv12.hbm`
