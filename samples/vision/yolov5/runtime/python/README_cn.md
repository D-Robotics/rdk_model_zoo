简体中文 | [English](./README.md)

# YOLOv5 运行说明

本目录提供 `yolov5` sample 在 `RDK X5` 上的 Python 推理入口。

## 文件说明

- `main.py`：统一 Python 推理入口。
- `yolov5_det.py`：基于 `hbm_runtime` 的 YOLOv5 检测封装。
- `run.sh`：一键运行脚本。

## 快速开始

```bash
cd runtime/python
chmod +x run.sh
./run.sh
```

脚本会在缺少默认模型时自动下载 `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` 到 `../../model/`，并将结果图保存到 `../../test_data/result.jpg`。

## 手动运行

```bash
python3 main.py
python3 main.py --model-path ../../model/yolov5s_tag_v7.0_detect_640x640_bayese_nv12.bin
python3 main.py --img-save-path ../../test_data/result_custom.jpg
```

## 命令行参数

```bash
python3 main.py -h
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-path` | BPU 量化 YOLOv5 BIN 模型路径。 | `../../model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` |
| `--label-file` | 可视化使用的类别名称文件路径。 | `../../../../../datasets/coco/coco_classes.names` |
| `--priority` | 运行时调度优先级。 | `0` |
| `--bpu-cores` | 推理使用的 BPU Core 编号。 | `0` |
| `--test-img` | 测试输入图片路径。 | `../../test_data/bus.jpg` |
| `--img-save-path` | 结果图保存路径。 | `../../test_data/result.jpg` |
| `--resize-type` | Resize 策略（`0`：直接缩放，`1`：letterbox）。 | `0` |
| `--classes-num` | 检测类别数量。 | `80` |
| `--score-thres` | 候选框筛选阈值。 | `0.25` |
| `--nms-thres` | 非极大值抑制 IoU 阈值。 | `0.45` |
| `--anchors` | 逗号分隔的 YOLOv5 anchor 配置。 | `10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326` |
| `--strides` | 逗号分隔的检测头 stride 配置。 | `8,16,32` |
