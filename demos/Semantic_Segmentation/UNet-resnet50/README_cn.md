[English](./README.md) | 简体中文

# UNet-resnet50 模型说明

UNet-resnet50 是基于 UNet 架构、以 ResNet-50 为骨干网络的语义分割模型。它能够对输入图像的每个像素进行分类，实现像素级别的场景理解，适用于自动驾驶感知、医学影像分析、工业检测等领域。

## 算法介绍

UNet 采用经典的编码器-解码器架构：

- **编码器（Encoder）**：使用卷积和池化操作逐步提取特征并降低空间维度，捕获上下文信息。
- **解码器（Decoder）**：通过上采样和卷积操作恢复空间分辨率，实现精确定位。
- **跳跃连接（Skip Connections）**：将编码器中的高分辨率特征与解码器中的特征拼接，保留细节信息，提高分割精度。
- **ResNet-50 骨干网络**：借助残差连接提供强大的特征提取能力，使深层网络能够有效训练。

原始实现及训练细节请参考：

- [Pytorch-UNet](https://github.com/bubbliiiing/unet-pytorch)

## 目录结构

```bash
.
|-- conversion/          # 模型转换流程说明（ONNX → BIN）
|-- model/               # 模型文件与下载脚本
|-- runtime/             # 推理示例（Python）
|   -- python/
|-- evaluator/           # 模型评估相关内容
|-- test_data/           # 示例输入或推理结果
|-- README.md            # 当前模型总览说明（英文）
|-- README_cn.md         # 当前模型总览说明（中文）
```

## 快速体验

本示例已提供转换好的 BPU 模型，普通用户可以直接运行推理，无需执行模型转换。

### Python

```bash
cd runtime/python
bash run.sh
```

`run.sh` 一键运行脚本将自动完成：
- 定位或提示模型文件
- 使用默认测试数据运行推理

详细说明请参考 [runtime/python/README.md](./runtime/python/README.md)。

## 模型转换

已提供转换好的模型，普通用户可以跳过此步骤。如需从 ONNX 重新转换或自行训练模型，请参考转换文档：

[conversion/README.md](./conversion/README.md)

## 模型推理

本示例同时提供 Python 推理实现。

- **Python**：详细的环境配置、参数说明与运行方式见 [runtime/python/README.md](./runtime/python/README.md)。

## 模型评估

关于评估指标（mIoU、像素准确率等）及评估脚本，请参考：

[evaluator/README.md](./evaluator/README.md)

## 推理结果

以下是 UNet-resnet50 在示例图像上的分割效果：

![](test_data/UNet_Segmentation_Origin.png)

*输入图像与带类别叠加的分割可视化结果。*

## 性能数据

### RDK X5 & RDK X5 Module

语义分割 Semantic Segmentation (VOC)

| 模型 | 尺寸(像素) | 类别数 | 参数量 | 吞吐量(单线程) <br/> 吞吐量(多线程) | 后处理时间(Python) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| UNet-resnet50 | 512×512 | 20 | 43.93 M | 11.23 FPS (1 thread) <br/> 13.23 FPS (2 threads) <br/> 13.23 (8 threads) | 267.08 ms |

### RDK X3 & RDK X3 Module

语义分割 Semantic Segmentation (VOC)

| 模型 | 尺寸(像素) | 类别数 | 参数量 | 吞吐量(单线程) <br/> 吞吐量(多线程) | 后处理时间(Python) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| UNet-resnet50 | 512×512 | 20 | 43.93 M | 2.61 FPS (1 thread) <br/> 5.17 FPS (2 threads) <br/> 5.21 FPS (4 threads) | 361.96 ms |

测试板卡均为最佳状态：

- **X5 最佳状态**：CPU 为 8 × A55@1.8G，全核心 Performance 调度，BPU 为 1 × Bayes-e@10TOPS。
- **X3 最佳状态**：CPU 为 4 × A53@1.8G，全核心 Performance 调度，BPU 为 2 × Bernoulli2@5TOPS。

关于后处理：目前在 X5 上使用 Python 实现的后处理（包括 Softmax 和 Argmax），单核心单线程约 3-5 ms 即可完成，后处理不会构成瓶颈。

## License

本模型示例遵循 ModelZoo 顶层开源协议，具体请参考仓库根目录的 LICENSE 文件。
