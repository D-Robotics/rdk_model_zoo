[English](./README.md) | 简体中文

# GoogLeNet 图像分类 Python 示例

本示例展示如何在 BPU 上使用量化后的 GoogLeNet 模型执行 ImageNet-1k 图像分类任务。

## 目录结构

```text
.
|-- main.py
|-- googlenet.py
|-- README.md
|-- README_cn.md
`-- run.sh
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-path` | 量化 `.bin` 模型文件路径。 | `/opt/hobot/model/x5/basic/googlenet_224x224_nv12.bin` |
| `--label-file` | ImageNet 标签文件路径。 | `../../test_data/ImageNet_1k.json` |
| `--priority` | 模型优先级，范围 `0~255`。 | `0` |
| `--bpu-cores` | 用于推理的 BPU 核索引。 | `0` |
| `--test-img` | 测试输入图像路径。 | `../../test_data/indigo_bunting.JPEG` |
| `--img-save-path` | 输出可视化图像保存路径。 | `../../test_data/result.jpg` |
| `--resize-type` | 缩放策略，`0` 为直接缩放，`1` 为 letterbox。 | `1` |
| `--topk` | 显示的 Top-K 类别数量。 | `5` |

## 快速运行

```bash
chmod +x run.sh
./run.sh
```

## 手动运行

- 使用默认参数：

```bash
python3 main.py
```

- 显式指定参数：

```bash
python3 main.py \
    --model-path ../../model/googlenet_224x224_nv12.bin \
    --test-img ../../test_data/indigo_bunting.JPEG \
    --img-save-path ../../test_data/result.jpg \
    --topk 5
```

## 接口说明

- **GoogLeNetConfig**：封装模型路径、标签文件和推理参数。
- **GoogLeNet**：实现预处理、BPU 推理和 Top-K 分类后处理。
