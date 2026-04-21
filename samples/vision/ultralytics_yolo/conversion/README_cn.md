[English](./README.md) | 简体中文

# Ultralytics YOLO 模型转换说明

本目录提供将 Ultralytics YOLO 模型转换为适用于 RDK X5 的量化 `.bin`
模型所需的工具和说明。

## 模型编译环境

模型转换需要在 x86 Linux 机器上准备 RDK X5 OpenExplore 工具链。

### 方式一：Pip 安装

```bash
conda create -n rdk_env python=3.10 -y
conda activate rdk_env
pip install rdkx5-yolo-mapper
hb_mapper --version
```

### 方式二：Docker 安装

```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
docker run -it --rm -v /path/to/rdk_model_zoo:/data openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

或者前往地瓜开发者社区获取离线版本的 Docker 镜像: [https://forum.d-robotics.cc/t/topic/28035](https://forum.d-robotics.cc/t/topic/28035)

## 转换流程

### 1. 导出 ONNX

在 Ultralytics 环境中使用 `export_monkey_patch.py` 将 `pt` 模型导出为
ONNX。

```bash
python3 export_monkey_patch.py --pt yolo11n.pt
```

### 2. 准备校准数据和 mapper 配置

使用 `mapper.py` 准备校准数据、生成 mapper 配置并调用 `hb_mapper`。

```bash
python3 mapper.py --onnx yolo11n.onnx --cal-images /path/to/calibration_images
```

### 3. 检查并编译 BIN 模型

```bash
hb_mapper checker --model-type onnx --config config.yaml
hb_mapper makertbin --config config.yaml
```

### 4. 检查编译结果

```bash
hb_perf config.yaml
hrt_model_exec model_info --model_file yolo11n_detect_bayese_640x640_nv12.bin
hrt_model_exec perf --model_file yolo11n_detect_bayese_640x640_nv12.bin --thread_num 1
```

## 输出 Tensor 协议

本 sample 的 Python runtime 使用固定输出协议进行解析，因此转换侧需要保持
以下协议不变。

### Detection

支持模型族：

- `YOLOv5u`
- `YOLOv8`
- `YOLOv9`
- `YOLOv10`
- `YOLO11`
- `YOLO12`
- `YOLO13`

输出顺序：

- `output[0]`: stride `8` 的分类输出
- `output[1]`: stride `8` 的 DFL box 输出
- `output[2]`: stride `16` 的分类输出
- `output[3]`: stride `16` 的 DFL box 输出
- `output[4]`: stride `32` 的分类输出
- `output[5]`: stride `32` 的 DFL box 输出

### Instance Segmentation

支持模型族：

- `YOLOv8`
- `YOLOv9`
- `YOLO11`

输出顺序：

- `[cls, box, mask_coeff] * 3`
- 最后一层为 `proto` 输出

### Pose Estimation

支持模型族：

- `YOLOv8`
- `YOLO11`

输出顺序：

- `[cls, box, keypoints] * 3`

### Classification

支持模型族：

- `YOLOv8`
- `YOLO11`

输出顺序：

- `output[0]`: `(1, 1000, 1, 1)`

## 参考日志

本目录保留了支持模型族的 `hb_mapper` 和 `hrt_model_exec` 参考日志，用于确认
输出 tensor 协议和转换结果。
