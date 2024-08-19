[English](./README.md) | 简体中文

# Transformer - EdgeNeXt

## 目录

- [1. 模型简介](#1-简介)
- [2. 模型性能数据](#2-模型性能数据)
- [3. 模型下载](#3-模型下载)
- [4. 部署测试](#4-部署测试)
- [5. 量化实验](#5-量化实验)

## 1. 简介

- 论文地址: [EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications](https://arxiv.org/abs/2206.10589)

- Github 仓库: [EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt)

![](./data/EdgeNeXt_architecture.png)

EdgeNeXt 网络从轻量化角度出发，设计了卷积与transformer的混合架构EdgeNeXt，兼顾了模型性能与模型大小/推理速度。

整体架构采取标准的“四阶段”金字塔范式设计，其中包含**卷积编码器**与**SDTA编码器**两个重要的模块。在卷积编码器中，自适应核大小的设计被应用，这与SDTA中的多尺度感受野的思想相呼应。而在SDTA编码器中，特征编码部分使用固定的3×3卷积，但通过层次级联实现多尺度感受野的融合，而此处若使用不同尺寸的卷积核是否会带来更好的效果有待考证。

在自注意计算部分，通过将点积运算应用于通道维度，得到了兼顾计算复杂度与全局注意力的输出，是支撑网络的一个核心点。从分类性能来看，效果确实很好，但结合检测、分割任务来看，供对比模型略少，仅提供了部分轻量级网络对比。

**EdgeNeXt 模型特点**：

- Transformer 与 CNN 的混合结构，在保持 Transformer 的精度同时有 CNN 的推理速度；
- 四段式结构金字塔，量化部署更加友好
- 通过通道组分割与合理的注意力机制增加感受野并编码多尺度特征，来提高资源利用率

## 2. 模型性能数据

以下表格是在 RDK X5 & RDK X5 Module 上实际测试得到的性能数据，可以根据自己推理实际需要的性能和精度，对模型的大小做权衡取舍


| 模型 | 尺寸(像素)  | 类别数  | 参数量(M) | 浮点精度  | 量化精度  | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) |
| ----------------- | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- |
| EdgeNeXt_base     | 224x224 | 1000 | 18.51  | 97.02 | 98.89 | 8.80        | 32.31       |
| EdgeNeXt_small    | 224x224 | 1000 | 5.59   | 93.30 | 91.31 | 4.41        | 14.93       |
| EdgeNeXt_x_small  | 224x224 | 1000 | 2.34   | 93.20 | 91.57 | 2.88        | 9.63        |
| EdgeNeXt_xx_small | 224x224 | 1000 | 1.33   | 84.29 | 84.15 | 2.47        | 7.24        |

说明: 
1. X5的状态为最佳状态：CPU为8xA55@1.8G, 全核心Performance调度, BPU为1xBayes-e@1G, 共10TOPS等效int8算力。
2. 单线程延迟为单帧，单线程，单BPU核心的延迟，BPU推理一个任务最理想的情况。
3. 4线程工程帧率为4个线程同时向双核心BPU塞任务，一般工程中4个线程可以控制单帧延迟较小，同时吃满所有BPU到100%，在吞吐量(FPS)和帧延迟间得到一个较好的平衡。
4. 8线程极限帧率为8个线程同时向X3的双核心BPU塞任务，目的是为了测试BPU的极限性能，一般来说4核心已经占满，如果8线程比4线程还要好很多，说明模型结构需要提高"计算/访存"比，或者编译时选择优化DDR带宽。
5. 浮点/定点mAP：浮点精度使用的是模型未量化前onnx的推理置信度，量化精度则为量化后模型实际推理的置信度。

## 3. 模型下载

**.bin 文件下载**：

可以使用脚本 [download_bin.sh](./model/download_bin.sh) 一键下载所有此模型结构的 .bin 模型文件，方便直接更换模型。或者使用以下命令行中的一个，选取单个模型进行下载：

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EdgeNeXt_base-deploy_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EdgeNeXt_small-deploy_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EdgeNeXt_x_small-deploy_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EdgeNeXt_xx_small-deploy_224x224_nv12.bin
```

**ONNX文件下载**：

与.bin文件同理，使用 [download_onnx.sh](./model/download_onnx.sh)一键下载所有此模型结构的 .onnx 模型文件，或下载单个 .onnx 模型进行量化实验：
```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/edgenext_base_deploy.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/edgenext_small_deploy.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/edgenext_x_small_deploy.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/edgenext_xx_small_deploy.onnx
```

## 4. 部署测试

在下载完毕 .bin 文件后，可以执行 test_EdgeNeXt_*.ipynb 系列的 EdgeNeXt 模型 jupyter 脚本文件，在板端实际运行体验实际测试效果。需要更改测试图片，可额外下载数据集后，放入到data文件夹下并更改 jupyter 文件中图片的路径

![alt text](./data/inference.png)

## 5. 量化实验

若想要进一步进阶对模型量化过程中的学习，如选取量化精度、对模型节点进行取舍、模型输入输出格式配置等，可以按顺序在天工开物工具链（注意是在pc端，不是板端）中执行 mapper 文件夹下的shell文件，对模型进行量化调优。

EdgeNeXt 由于内部带有 softmax 结点，天工开物工具链默认将 softmax 结点放在CPU上执行，需要在yaml配置文件中的 model_parameters 参数下的 node_info 将 softmax 在BPU中进行量化。这里仅仅给出 yaml 的配置文件（在yaml文件夹中），如需进行量化实验可将对应不同大小模型的yaml文件自行替换