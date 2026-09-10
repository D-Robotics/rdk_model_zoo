# UNet-resnet50 Python 推理

[English](./README.md) | 简体中文

本目录包含 UNet-resnet50 语义分割模型的 Python 推理示例。

## 简介

UNet-resnet50 模型执行像素级语义分割，为输入图像中的每个像素预测类别标签。本示例演示如何加载 BPU 量化模型、预处理图像、运行推理并可视化分割结果。

## 环境依赖

- Python 3.8+
- OpenCV（`cv2`）
- NumPy
- Horizon `hobot_dnn`（pyeasy_dnn）

除标准 Horizon 运行时环境外，无需额外的 Python 包。

## 目录结构

```bash
.
|-- unet.py      # 模型实现（Config + Model 类）
|-- main.py      # 主推理入口
|-- run.sh       # 一键运行脚本
`-- README.md    # 本文件
```

## 参数说明

| 参数           | 说明                                                 | 默认值                                                     |
| -------------- | ---------------------------------------------------- | ---------------------------------------------------------- |
| `--model-path` | BPU 量化 *.bin 模型文件路径                          | `./model/unet_resnet50_512x512_nv12.bin`               |
| `--img-path`   | 待推理的输入图像路径                                 | `./test_data/1.jpg`             |
| `--save-path`  | 可视化结果的保存路径                                 | `unet_result.jpg`                                          |
| `--mask-path`  | 原始分割掩码的保存路径                               | `unet_mask.png`                                            |
| `--num-classes`| 分割类别数（含背景）                                 | `21`                                                       |
| `--alpha`      | 可视化叠加掩码的不透明度                             | `0.6`                                                      |

## 快速开始

### 1. 默认参数运行（零配置）

若模型已放置在系统默认路径且测试数据存在：

```bash
bash run.sh
```

或直接运行：

```bash
python3 main.py
```

### 2. 自定义参数运行

```bash
python3 main.py \
    --model-path /path/to/unet_resnet50_512x512_nv12.bin \
    --img-path /path/to/image.jpg \
    --save-path result.jpg \
    --num-classes 21
```

### 输出

- `unet_result.jpg` —— 带颜色叠加与图例的可视化结果
- `unet_mask.png` —— 原始分割掩码（类别索引放大后便于观察）

## 代码文档

源码的详细 API 文档请参考 `docs/source_reference` 目录中的源码参考文档。

## 注意事项

- 输入图像必须是有效的 3 通道 BGR 图像。
- 模型期望的输入为分辨率 512×512 的 NV12 格式。
- 分割掩码使用最近邻插值缩放回原始图像尺寸，以保留类别边界。
