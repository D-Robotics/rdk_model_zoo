[English](./README.md) | [简体中文](./README_cn.md)

# UNet 模型说明

本示例提供基于 ResNet18、ResNet34、ResNet50、ResNet101 和 ResNet152 主干的
UNet Pascal VOC 语义分割部署链路，覆盖 checkpoint 导出、X5 PTQ 转换、精度
评测和 RDK X5 Python 推理。

## 算法介绍（Algorithm Overview）

UNet 使用带跳跃连接的编码器—解码器结构，融合高层语义与精细空间信息。本实现
以 ResNet 作为编码器，以 UNet 解码器为每个像素输出类别得分。

- 任务：Pascal VOC 21 类语义分割，包含背景类
- UNet 论文：[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- ResNet 论文：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- 参考实现：[bubbliiiing/unet-pytorch](https://github.com/bubbliiiing/unet-pytorch)

### 部署合同

| 项目 | 约定 |
| --- | --- |
| 目标平台 | RDK X5，`bayes-e` |
| 训练输入 | RGB float32 NCHW `[1, 3, 512, 512]`，缩放系数 `1/255` |
| Runtime 输入 | 512 × 512 packed NV12 |
| 输出 | float32 NCHW logits `[1, 21, 512, 512]` |
| 后处理 | 沿类别维执行 `argmax` |

### 主干支持状态

| Backbone | 当前状态 |
| --- | --- |
| ResNet18 | 预编译 X5 BIN 已发布；已在 RDK X5 验证下载与 Python 推理 |
| ResNet34 | FP32 mIoU 0.689319；ONNX 导出与 X5 PTQ 通过；BIN 已发布，板端 Runtime 待测 |
| ResNet50 | FP32 mIoU 0.683826；ONNX 导出与 X5 PTQ 通过；BIN 已发布，板端 Runtime 待测 |
| ResNet101 | FP32 mIoU 0.709437；ONNX 导出与 X5 PTQ 通过；BIN 已发布，板端 Runtime 待测 |
| ResNet152 | FP32 mIoU 0.740002；ONNX 导出与 X5 PTQ 通过；BIN 已发布，板端 Runtime 待测 |

上游 ResNet50 VOC checkpoint 可从
[`unet-pytorch` v1.0 release](https://github.com/bubbliiiing/unet-pytorch/releases/download/v1.0/unet_resnet_voc.pth)
下载，SHA256 为
`556a74b8379c40cbc76af7a1faab84d1316f02b7d93290b5f1f724ff922faacb`。
新训练变体使用 torchvision ImageNet encoder 初始化；训练收据必须记录准确的
torchvision 版本与权重标识。

共享模型源码和转换模板不代表未经测试的 backbone 已经受支持。每个变体都必须
分别通过 checkpoint、ONNX、PTQ、精度、Runtime 和板端性能门禁。

## 目录结构（Directory Structure）

```text
unet/
├── conversion/                         # checkpoint 到 X5 的转换流程
│   ├── mapper.py                       # 带门禁的 checker/makertbin 入口
│   ├── onnx_export/
│   │   ├── export_unet.py              # 严格加载与 ONNX 导出入口
│   │   └── model/                      # 共享 UNet ResNet 模型定义
│   ├── ptq_yamls/                      # 每个 backbone 一份 bayes-e 模板
│   ├── README.md
│   └── README_cn.md
├── evaluator/                          # PyTorch/ONNX/X5 统一精度入口
│   ├── eval_unet.py
│   ├── README.md
│   └── README_cn.md
├── model/                              # 预编译 X5 模型与下载说明
│   ├── download_model.sh               # 按 backbone 下载模型
│   ├── README.md
│   └── README_cn.md
├── runtime/
│   └── python/                         # RDK X5 hbm_runtime 示例
│       ├── unet.py                     # UNetConfig 与 UNet 封装
│       ├── main.py                     # 命令行推理入口
│       ├── run.sh                      # 一键运行脚本
│       ├── README.md
│       └── README_cn.md
├── test_data/                          # 默认 Pascal VOC 测试图片
│   ├── 2007_000033.jpg
│   ├── README.md
│   └── README_cn.md
├── README.md
└── README_cn.md
```

训练工具与生成的中间产物维护在本示例之外。仓库不提交 checkpoint、ONNX、
校准数据、编译后的 BIN 或完整评测数据集；预编译 BIN 通过 `model/` 中的脚本下载。

## 快速体验（QuickStart）

Python Runtime 是面向用户的推理入口。直接零参数运行；默认 ResNet18 不存在时，
启动脚本会自动下载：

```bash
cd samples/vision/unet/runtime/python
./run.sh
```

该命令加载 X5 BIN，把 BGR 图片转换为 packed NV12，执行 BPU 推理，并保存类别
索引 mask、彩色叠加图和 JSON 报告。参数与接口说明见
[Python Runtime 文档](./runtime/python/README_cn.md)。

## 模型转换（Model Conversion）

仓库已经提供五个 backbone 的预编译模型，普通用户可以跳过转换。需要复现 checkpoint
导出、校准、checker 和 makertbin 的开发者请阅读[转换文档](./conversion/README_cn.md)。

## 模型推理（Runtime）

本示例当前提供基于 `hbm_runtime` 的 Python 实现。环境配置、默认路径、CLI 参数、
输出文件和可复用接口见
[runtime/python/README_cn.md](./runtime/python/README_cn.md)。

## 模型评估

统一 evaluator 可以使用同一份 Pascal VOC manifest 评测 PyTorch checkpoint、
ONNX 或 X5 BIN，详见[evaluator/README_cn.md](./evaluator/README_cn.md)。

## 参考结果

以下结果来自先前用于验证部署链路的 ResNet18 基线 checkpoint，在完整 1,449 张
Pascal VOC 验证集上测得。它们不是本次下载模型的重新评测结果，也不能代表其他
backbone。维护者后续已在 RDK X5 上确认已发布 ResNet18 可以正常下载并输出 mask
与 overlay；板端精度和纯 BPU 性能复测仍待补。

| 后端 | mIoU | Pixel Accuracy |
| --- | ---: | ---: |
| PyTorch FP32 | 0.619695 | 0.911532 |
| ONNX Runtime FP32 | 0.619694 | 0.911532 |
| RDK X5 PTQ | 0.617198 | 0.910332 |

RDK X5 上使用单线程、200 帧和真实 packed NV12 输入时，`hrt_model_exec` 实测
平均延迟 52.72 ms、18.96 FPS。Python 端到端延迟还包含图片解码、预处理、
输出回读和后处理，不能与纯 Runtime 指标直接等同。

### ResNet34/50/101/152 发布结果

四个新训练变体均在同一份完整 1,449 张验证集上评测。它们通过了 ONNX 数值门禁和
`bayes-e` PTQ 编译，公开 BIN 也已重新下载并与发布 SHA256 核对一致。本次没有为
这四个发布模型执行板端 Runtime 精度和性能测试。

| Backbone | PyTorch FP32 mIoU | Pixel Accuracy | PTQ 输出 Cosine | 板端 Runtime |
| --- | ---: | ---: | ---: | --- |
| ResNet34 | 0.689319 | 0.930947 | 0.998328 | 待测 |
| ResNet50 | 0.683826 | 0.928404 | 0.995292 | 待测 |
| ResNet101 | 0.709437 | 0.935887 | 0.996384 | 待测 |
| ResNet152 | 0.740002 | 0.942805 | 0.996070 | 待测 |

ResNet50 相较上游 checkpoint 的 0.661483 mIoU 有提升，但没有超过 ResNet34。
完整系列的下载地址和 SHA256 见[模型下载说明](./model/README_cn.md)。

## License

本示例遵循仓库顶层 [Apache License 2.0](../../../LICENSE)。参考 UNet 实现派生自
[`bubbliiiing/unet-pytorch`](https://github.com/bubbliiiing/unet-pytorch) 的
commit `5bcd6b4c832648beed1b92e78ed1e85c56343eca`，并保留其 MIT 条款。

<details>
<summary>UNet 派生代码的 MIT 声明</summary>

```text
MIT License

Copyright (c) 2021 Bubbliiiing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</details>
