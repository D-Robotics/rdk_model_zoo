[English](./README.md) | 简体中文

# Transformer - ConvNeXt

- [Transformer - ConvNeXt](#transformer---convnext)
  - [1. 简介](#1-简介)
  - [2. 模型性能数据](#2-模型性能数据)
  - [3. 模型下载](#3-模型下载)
  - [4. 部署测试](#4-部署测试)
  - [5. 量化实验](#5-量化实验)

## 1. 简介

- **论文地址**: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

- **Github 仓库**: [facebookresearch/ConvNeXt: Code release for ConvNeXt model (github.com)](https://github.com/facebookresearch/ConvNeXt)

ConvNeXt 网络从原始的 ResNet 出发，通过借鉴 Swin Transformer 的设计来逐步地改进模型。ConvNeXt并没有特别复杂或者创新的结构，它的每一个网络细节都是已经在不止一个网络中被采用，通过堆叠各种深度模型结构设计的技术达到sota的效果。网络结构中使用了更大的卷积核（7x7）、ReLU替换为GELU激活函数、更少的激活函数、LayerNorm 替代 BatchNorm，以及减少了下采样的频率。这些改动让 ConvNeXt 具有较好的表达能力，同时保留了卷积网络的高效性，其设计简单且高效，适用于各种视觉任务。

![](./data/ConvNeXt_Block.png)

**ConvNeXt 模型特点**：

- **大核卷积**：ConvNeXt 使用了大核 (7x7) 卷积代替传统的 3x3 卷积。大核卷积有助于扩大感受野，同时减少网络深度的需求
- **深度可分离卷积**：ConvNeXt 引入了深度可分离卷积，类似于 [MobileNet](../MobileNetV1/README_cn.md) 和 [EfficientNet](../EfficientNet/README_cn.md),这样可以显著减少参数量和计算成本
- **归一化层**：ConvNeXt 用 LayerNorm 替换了传统的 BatchNorm，这使得网络更适应于小批量数据，同时简化了模型设计
- **简化的残差连接**：ConvNeXt 简化了全连接层的设计，去掉了 ResNet 中的瓶颈结构，代之以更简单的卷积操作和全连接层

## 2. 模型性能数据

以下表格是在 RDK X5 & RDK X5 Module 上实际测试得到的性能数据，可以根据自己推理实际需要的性能和精度，对模型的大小做权衡取舍

| 模型 | 尺寸(像素)  | 类别数  | 参数量(M) | 浮点Top-1  | 量化Top-1  | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) | 帧率 |
| ----------------- | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------ |
| ConvNeXt_nano  | 224x224 | 1000 | 15.59  | 77.37 | 71.75 | 5.71        | 19.80       | 200.18 |
| ConvNeXt_pico  | 224x224 | 1000 | 9.04   | 77.25 | 71.03 | 3.37        | 10.88       | 364.07 |
| ConvNeXt_femto | 224x224 | 1000 | 5.22   | 73.75 | 72.25 | 2.46        | 7.11        | 556.02 |
| ConvNeXt_atto  | 224x224 | 1000 | 3.69   | 73.25 | 69.75 | 1.96        | 5.39        | 732.10 |


说明: 
1. X5的状态为最佳状态：CPU为8xA55@1.8G, 全核心Performance调度, BPU为1xBayes-e@1G, 共10TOPS等效int8算力。
2. 单线程延迟为单帧，单线程，单BPU核心的延迟，BPU推理一个任务最理想的情况。
3. 4线程工程帧率为4个线程同时向双核心BPU塞任务，一般工程中4个线程可以控制单帧延迟较小，同时吃满所有BPU到100%，在吞吐量(FPS)和帧延迟间得到一个较好的平衡。
4. 8线程极限帧率为8个线程同时向X3的双核心BPU塞任务，目的是为了测试BPU的极限性能，一般来说4核心已经占满，如果8线程比4线程还要好很多，说明模型结构需要提高"计算/访存"比，或者编译时选择优化DDR带宽。
5. 浮点/定点Top-1：浮点Top-1使用的是模型未量化前onnx的 Top-1 推理精度，量化Top-1则为量化后模型实际推理的精度。

## 3. 模型下载

**.bin 文件下载**：

可以使用脚本 [download_bin.sh](./model/download_bin.sh) 一键下载所有此模型结构的 .bin 模型文件，方便直接更换模型。或者使用以下命令行中的一个，选取单个模型进行下载：

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ConvNeXt_atto_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ConvNeXt_femto_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ConvNeXt_nano_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ConvNeXt_pico_224x224_nv12.bin
```

**ONNX文件下载**：

与.bin文件同理，使用 [download_onnx.sh](./model/download_onnx.sh)一键下载所有此模型结构的 .onnx 模型文件，或下载单个 .onnx 模型进行量化实验：
```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ConvNeXt_atto.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ConvNeXt_femto.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ConvNeXt_nano.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ConvNeXt_pico.onnx
```

## 4. 部署测试

在下载完毕 .bin 文件后，可以执行 test_ConvNeXt_*.ipynb 系列的 ConvNeXt 模型 jupyter 脚本文件，在板端实际运行体验实际测试效果。需要更改测试图片，可额外下载数据集后，放入到data文件夹下并更改 jupyter 文件中图片的路径

![](./data/inference.png)

## 5. 量化实验

若想要进一步进阶对模型量化过程中的学习，如选取量化精度、对模型节点进行取舍、模型输入输出格式配置等，可以按顺序在天工开物工具链（注意是在pc端，不是板端）中执行 mapper 文件夹下的shell文件，对模型进行量化调优。这里仅仅给出 yaml 的配置文件（在yaml文件夹中），如需进行量化实验可将对应不同大小模型的yaml文件自行替换