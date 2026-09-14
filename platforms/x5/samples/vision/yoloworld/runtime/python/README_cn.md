[English](./README.md) | 简体中文

# YOLOWorld Python 示例

本示例说明如何使用 YOLOWorld 和离线词表 embedding 完成开放词表目标检测。

## 目录结构

```text
.
|-- main.py
|-- yoloworld_det.py
|-- README.md
|-- README_cn.md
`-- run.sh
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-path` | 量化 `.bin` 模型路径。 | `../../model/yolo_world.bin` |
| `--vocab-file` | 离线词表 embedding JSON 路径。 | `../../test_data/offline_vocabulary_embeddings.json` |
| `--test-img` | 测试输入图片路径。 | `../../test_data/dog.jpeg` |
| `--prompts` | 逗号分隔的检测 prompt 词。 | `dog` |
| `--img-save-path` | 可视化结果图保存路径。 | `../../test_data/inference.png` |
| `--priority` | 模型优先级，范围 `0~255`。 | `0` |
| `--bpu-cores` | 推理使用的 BPU 核心索引。 | `0` |
| `--score-thres` | 预测结果过滤分数阈值。 | `0.05` |
| `--nms-thres` | NMS 重叠框抑制阈值。 | `0.45` |

## 快速运行

```bash
chmod +x run.sh
./run.sh
```

## 手动运行

直接运行 `main.py` 前需要先准备默认模型。可以先执行一次 `./run.sh`，也可以在本目录下执行
`../../model/download.sh` 下载 `../../model/yolo_world.bin`。

- 使用默认参数运行：

```bash
python3 main.py
```

- 使用显式参数运行：

```bash
python3 main.py \
    --model-path ../../model/yolo_world.bin \
    --vocab-file ../../test_data/offline_vocabulary_embeddings.json \
    --test-img ../../test_data/dog.jpeg \
    --prompts dog \
    --img-save-path ../../test_data/inference.png
```

## 接口说明

- **YOLOWorldConfig**：封装模型路径、词表路径和阈值。
- **YOLOWorldDetect**：实现图像和文本 embedding 预处理、BPU 推理和固定 YOLOWorld 输出解析。
