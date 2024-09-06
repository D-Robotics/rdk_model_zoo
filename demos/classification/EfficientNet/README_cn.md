[English](./README.md) | 简体中文

# CNN X5 - EfficientNet

- [CNN X5 - EfficientNet](#cnn-x5---efficientnet)
  - [1. 简介](#1-简介)
  - [2. 模型性能数据](#2-模型性能数据)
  - [3. 模型下载](#3-模型下载)
  - [4. 部署测试](#4-部署测试)
  - [5. 量化实验](#5-量化实验)

## 1. 简介

- **论文地址**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

- **Github 仓库**: [lukemelas/EfficientNet-PyTorch: A PyTorch implementation of EfficientNet (github.com)](https://github.com/lukemelas/EfficientNet-PyTorch)

![](./data/EfficientNet_architecture.png)

EfficientNet 是一种平衡卷积神经网络（CNN）分辨率、深度和宽度的创新网络结构。其核心特点在于通过复合缩放（Compound Scaling）方法，系统性地优化网络性能与效率。传统的网络设计通常只对网络的单一维度进行调整，例如宽度、深度或输入图像的分辨率。而 EfficientNet 则通过一个简单且高效的复合系数，统一调整这三个关键维度。这种方法通过网格搜索来确定每个维度的最佳比例，从而在给定的资源限制下，最大化网络的整体性能。

EfficientNet 结合了 AutoML 技术，自动搜索出最佳的网络参数组合，使模型在保持高准确率的同时，大幅减少参数量和计算资源。例如，EfficientNet-B7 在 ImageNet 数据集上实现了 84.3% 的 top-1 准确率，且其参数量仅为 GPipe 的 1/8.4，推理速度提升了 6.1 倍。

**EfficientNet 模型特点**：

- **复合缩放方法**：通过统一调整分辨率、深度和宽度三个维度，最大化模型性能与效率，而不是单独调整某一个维度。
- **AutoML 技术**：利用 AutoML 自动搜索最佳网络参数组合，使模型在保持高准确率的同时，显著减少参数量和计算资源消耗。
- **高效且轻量化的网络结构**：通过优化的复合缩放策略，EfficientNet 实现了较少的参数量和更快的推理速度，但在处理较大模型时可能会消耗更多显存。


## 2. 模型性能数据

以下表格是在 RDK X5 & RDK X5 Module 上实际测试得到的性能数据，可以根据自己推理实际需要的性能和精度，对模型的大小做权衡取舍。


| 模型           | 尺寸(像素)  | 类别数  | 参数量(M) | 浮点Top-1  | 量化Top-1  | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) | 帧率      |
| ------------ | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------- |
| Efficientnet_B4   | 224x224     | 1000     | 19.27     | 74.25     | 71.75     | 5.44        | 18.63       | 212.75      |
| Efficientnet_B3   | 224x224     | 1000     | 12.19     | 76.22     | 74.05     | 3.96        | 12.76       | 310.30      |
| Efficientnet_B2   | 224x224     | 1000     | 9.07      | 76.50     | 73.25     | 3.31        | 10.51       | 376.77      |


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
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EfficientNet_B2_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EfficientNet_B3_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EfficientNet_B4_224x224_nv12.bin
```

**ONNX文件下载**：

与.bin文件同理，使用 [download_onnx.sh](./model/download_onnx.sh)一键下载所有此模型结构的 .onnx 模型文件，或下载单个 .onnx 模型进行量化实验：

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EfficientNet_B2_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EfficientNet_B3_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EfficientNet_B4_224x224_nv12.bin
```

## 4. 部署测试

在下载完毕 .bin 文件后，可以执行 test_EfficientNet_*.ipynb 系列的 EfficientNet 模型 jupyter 脚本文件，在板端实际运行体验实际测试效果。需要更改测试图片，可额外下载数据集后，放入到data文件夹下并更改 jupyter 文件中图片的路径

![](./data/inference.png)

## 5. 量化实验

若想要进一步进阶对模型量化过程中的学习，如选取量化精度、对模型节点进行取舍、模型输入输出格式配置等，可以按顺序在天工开物工具链（注意是在pc端，不是板端）中执行 mapper 文件夹下的shell文件，对模型进行量化调优。这里仅仅给出 yaml 的配置文件（在yaml文件夹中），如需进行量化实验可将对应不同大小模型的yaml文件自行替换
