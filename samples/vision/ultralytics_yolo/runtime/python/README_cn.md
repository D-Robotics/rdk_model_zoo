[English](./README.md) | 简体中文

# Ultralytics YOLO Python Sample

本 sample 展示了如何在 RDK X5 上通过 `hbm_runtime` 运行 Ultralytics YOLO
各任务模型。

## 环境依赖

本 sample 没有额外的特殊依赖，请确保 RDK X5 Python 环境已经准备完成。

```bash
pip install numpy opencv-python hbm-runtime scipy
```

## 目录结构

```text
.
|-- main.py                 # 推理入口脚本
|-- ultralytics_yolo_det.py # 检测封装
|-- ultralytics_yolo_seg.py # 分割封装
|-- ultralytics_yolo_pose.py# 姿态封装
|-- ultralytics_yolo_cls.py # 分类封装
|-- run.sh                  # 一键运行脚本
`-- README_cn.md            # 使用说明
```

## 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--task` | 任务类型：`detect`、`seg`、`pose`、`cls` | `detect` |
| `--model-path` | RDK X5 `.bin` 模型路径 | `../../model/yolo11n_detect_bayese_640x640_nv12.bin` |
| `--test-img` | 测试图片路径 | `../../test_data/bus.jpg` |
| `--label-file` | 标签文件路径，留空时使用任务默认值 | `""` |
| `--img-save-path` | `detect`、`seg`、`pose` 结果图保存路径 | `../../test_data/result_detect.jpg` |
| `--priority` | 模型优先级 | `0` |
| `--bpu-cores` | 推理使用的 BPU core 列表 | `[0]` |
| `--classes-num` | 检测类任务的类别数 | `80` |
| `--score-thres` | 置信度阈值 | `0.25` |
| `--nms-thres` | NMS 阈值 | `0.70` |
| `--strides` | 解码 stride | `8,16,32` |
| `--reg` | DFL 回归通道数 | `16` |
| `--mc` | 分割 mask coefficient 通道数 | `32` |
| `--nkpt` | 姿态关键点数量 | `17` |
| `--kpt-conf-thres` | 姿态关键点显示阈值 | `0.50` |
| `--topk` | 分类任务输出的 Top-K 数量 | `5` |
| `--resize-type` | 缩放策略，`0` 为直接 resize，`1` 为 letterbox | `1` |

## 快速运行

- **一键运行脚本**
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

- **手动运行**
  - 使用默认参数
    ```bash
    python3 main.py
    ```
  - 显式指定检测参数
    ```bash
    python3 main.py \
        --task detect \
        --model-path ../../model/yolo11n_detect_bayese_640x640_nv12.bin \
        --test-img ../../test_data/bus.jpg \
        --img-save-path ../../test_data/result_detect.jpg
    ```

## 任务示例

### 分割

```bash
python3 main.py \
    --task seg \
    --model-path ../../model/yolo11n_seg_bayese_640x640_nv12.bin \
    --test-img ../../test_data/bus.jpg \
    --img-save-path ../../test_data/result_seg.jpg
```

### 姿态

```bash
python3 main.py \
    --task pose \
    --model-path ../../model/yolo11n_pose_bayese_640x640_nv12.bin \
    --test-img ../../test_data/bus.jpg \
    --img-save-path ../../test_data/result_pose.jpg
```

### 分类

```bash
python3 main.py \
    --task cls \
    --model-path ../../model/yolo11n_cls_detect_bayese_640x640_nv12.bin \
    --test-img ../../test_data/zebra_cls.jpg \
    --label-file ../../../../../datasets/imagenet/imagenet_classes.names
```

## 接口说明

- **`UltralyticsYOLO*Config`**：封装各任务的模型路径和运行参数。
- **`UltralyticsYOLO*`**：提供完整推理流程，包括 `pre_process`、`forward`、`post_process` 和 `predict`。

公共预处理、后处理和可视化工具位于 `utils/py_utils/` 目录。
