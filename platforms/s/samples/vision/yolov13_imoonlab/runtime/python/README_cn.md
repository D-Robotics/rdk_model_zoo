[English](./README.md) | 简体中文

# YOLOv13 iMoonLab Python 运行时

本目录演示如何在 RDK S100 / S100P 上使用 `hbm_runtime` 运行 YOLOv13 Detect 模型。

## 环境依赖

```bash
pip3 install numpy==1.26.4 opencv-python==4.11.0.86 scipy==1.15.3
```

`hbm_runtime` 由板端系统环境提供。

## 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-path` | `.hbm` 模型路径 | `../../model/s100/yolo13n_detect_nashe_640x640_nv12.hbm` |
| `--priority` | 模型优先级 | `0` |
| `--bpu-cores` | BPU 核心索引列表 | `0` |
| `--test-img` | 测试图片路径 | `../../test_data/kite.jpg` |
| `--label-file` | 类别文件路径 | `../../test_data/coco_classes.names` |
| `--img-save-path` | 结果图保存路径 | `result.jpg` |
| `--nms-thres` | NMS 阈值 | `0.45` |
| `--score-thres` | 置信度阈值 | `0.25` |

## 快速运行

### 一键脚本

```bash
cd runtime/python
bash run.sh
```

### 直接运行 main.py

```bash
python3 main.py \
  --model-path ../../model/s100/yolo13n_detect_nashe_640x640_nv12.hbm \
  --test-img ../../test_data/kite.jpg \
  --label-file ../../test_data/coco_classes.names \
  --img-save-path result.jpg
```

## 运行流程

`main.py` 仅负责参数解析、输入读取、配置构造、`predict()` 调用和结果保存。`yolov13.py` 负责 `set_scheduling_params(...)`、`pre_process(...)`、`forward(...)`、`post_process(...)`、`predict(...)` 与 `__call__(...)`。

## 输入输出协议

### 输入

- `input[0]`: Y plane
- `input[1]`: UV plane

### 输出

- `output[0]`: small stride classification
- `output[1]`: small stride box distribution
- `output[2]`: medium stride classification
- `output[3]`: medium stride box distribution
- `output[4]`: large stride classification
- `output[5]`: large stride box distribution

## License

本目录内容遵循仓库顶层 `LICENSE`。
