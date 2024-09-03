[English](./README.md) | 简体中文

# CNN X5 - MobileNetV3

- [CNN X5 - MobileNetV3](#cnn-x5---mobilenetv3)
  - [1. 简介](#1-简介)
  - [2. 模型性能数据](#2-模型性能数据)
  - [3. 模型下载](#3-模型下载)
  - [4. 部署测试](#4-部署测试)
  - [5. 量化实验](#5-量化实验)

## 1. 简介

- **论文地址**: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

- **Github 仓库**: [pytorch-image-models/timm/models/mobilenetv3.py at main · huggingface/pytorch-image-models (github.com)](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv3.py)

![](./data/MobileNetV3_architecture.png)

MobileNetV3 是对 [MobileNetV2](../MobileNetV2/README_cn.md) 的改进，同样是一种轻量级的神经网络。MobileNetV3 通过使用**网络架构搜索(network architecture search，NAS)** 和 NetAdapt 对网络进行优化。同时，论文提出了 MobileNetV3-Large 和 MobileNetV3-Small 两种不同的网络应对于不同的实际使用情况。与 [MobileNetV2](../MobileNetV2/README_cn.md) 相比，MobileNetV3-Large在 ImageNet 数据集识别的准确率高3.2%，同时减少15%的延迟，而 MobileNetV3-Small 的准确率高4.6%，同时减少5%的延迟。MobileNetV3-Large 在 MS COCO 数据集的检测精度大致比 [MobileNetV2](../MobileNetV2/README_cn.md) 快25%

**MobileNetV3 模型特点**：

- **深度可分离卷积 (Depthwise Separable Convolution)**：MobileNetV3 继承了 MobileNetV2 的深度可分离卷积结构，将标准卷积分解为深度卷积和逐点卷积，大大减少了计算量和参数量
- **倒残差结构 (Inverted Residuals)**：与 [MobileNetV2](../MobileNetV2/README_cn.md) 类似，MobileNetV3 使用了倒残差块 (Inverted Residual Block)，其中包含扩展卷积、深度卷积和逐点卷积，并在输入和输出之间添加了跳跃连接 (Skip Connection)
- **Squeeze-and-Excitation (SE) 模块**：MobileNetV3 引入了 SE 模块，通过全局池化和通道注意力机制 (Channel Attention Mechanism) 来重新调整通道权重，以增强特征的表示能力
- **H-Swish 激活函数**：MobileNetV3 使用了一种新的激活函数 H-Swish (Hard-Swish)，这是一种硬化版本的 Swish 激活函数，能够在保持精度的同时减少计算复杂度
- **NAS (Neural Architecture Search)**：MobileNetV3 的架构部分是通过神经架构搜索 (NAS) 自动优化得到的，这种方法能够在不同设备的条件下找到性能和效率的平衡


## 2. 模型性能数据

以下表格是在 RDK X5 & RDK X5 Module 上实际测试得到的性能数据，可以根据自己推理实际需要的性能和精度，对模型的大小做权衡取舍。


| 模型           | 尺寸(像素)  | 类别数  | 参数量(M) | 浮点精度  | 量化精度  | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) | 帧率      |
| ------------ | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------- |
| Mobilenetv3_large_100   | 224x224 | 1000 | 5.47   | 74.75 | 64.75 | 2.02        | 5.53        | 714.22 |


说明: 
1. X5的状态为最佳状态：CPU为8xA55@1.8G, 全核心Performance调度, BPU为1xBayes-e@1G, 共10TOPS等效int8算力。
2. 单线程延迟为单帧，单线程，单BPU核心的延迟，BPU推理一个任务最理想的情况。
3. 4线程工程帧率为4个线程同时向双核心BPU塞任务，一般工程中4个线程可以控制单帧延迟较小，同时吃满所有BPU到100%，在吞吐量(FPS)和帧延迟间得到一个较好的平衡。
4. 8线程极限帧率为8个线程同时向X3的双核心BPU塞任务，目的是为了测试BPU的极限性能，一般来说4核心已经占满，如果8线程比4线程还要好很多，说明模型结构需要提高"计算/访存"比，或者编译时选择优化DDR带宽。
5. 浮点/定点精度：浮点精度使用的是模型未量化前onnx的 Top-1 推理精度，量化精度则为量化后模型实际推理的精度。

## 3. 模型下载

**.bin 文件下载**：

可以使用脚本 [download_bin.sh](./model/download_bin.sh) 一键下载所有此模型结构的 .bin 模型文件，方便直接更换模型。或者使用以下命令行中的一个，选取单个模型进行下载：

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/MobileNetV3_224x224_nv12.bin
```

**ONNX文件下载**：

与.bin文件同理，使用 [download_onnx.sh](./model/download_onnx.sh)一键下载所有此模型结构的 .onnx 模型文件，或下载单个 .onnx 模型进行量化实验：

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/mobilenetv3_large_100.onnx
```

## 4. 部署测试

在下载完毕 .bin 文件后，可以执行 test_MobileNetV3.ipynb 系列的 MobileNetV3 模型 jupyter 脚本文件，在板端实际运行体验实际测试效果。需要更改测试图片，可额外下载数据集后，放入到data文件夹下并更改 jupyter 文件中图片的路径

![](./data/inference.png)

## 5. 量化实验

若想要进一步进阶对模型量化过程中的学习，如选取量化精度、对模型节点进行取舍、模型输入输出格式配置等，可以按顺序在天工开物工具链（注意是在pc端，不是板端）中执行 mapper 文件夹下的shell文件，对模型进行量化调优。这里仅仅给出 yaml 的配置文件（在yaml文件夹中），如需进行量化实验可将对应不同大小模型的yaml文件自行替换
