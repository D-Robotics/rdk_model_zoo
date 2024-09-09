[English](./README.md) | 简体中文

# CNN - MobileOne

- [CNN - MobileOne](#cnn---mobileone)
  - [1. 简介](#1-简介)
  - [2. 模型性能数据](#2-模型性能数据)
  - [3. 模型下载](#3-模型下载)
  - [4. 部署测试](#4-部署测试)
  - [5. 量化实验](#5-量化实验)

## 1. 简介

- **论文地址**: [MobileOne: An Improved One millisecond Mobile Backbone](http://arxiv.org/abs/2206.04040)

- **Github 仓库**: [apple/ml-mobileone: This repository contains the official implementation of the research paper, "An Improved One millisecond Mobile Backbone". (github.com)](https://github.com/apple/ml-mobileone)

![](./data/MobileOne_architecture.png)

MobileOne 是一种借助了结构重参数化技术的，在端侧设备上很高效的视觉骨干架构（iPhone 12上MobileOne的推理时间只有1毫秒）。而且，与部署在移动设备上的现有架构相比，采用了**结构重参数化**的方法，为了加快推理速度没有加入常用的残差连接。MobileOne 可以推广到多个任务：图像分类、对象检测和语义分割，在延迟和准确性方面有显著改进。MobileOne的核心模块基于MobileNetV1而设计，同时吸收了重参数思想。采用的基本架构是 3x3 depthwise convolution + 1x1 pointwise convolution


**MobileOne 模型特点**：

- 在移动设备上(iphone12)可以在1ms内运行，且与其他有效/轻量级网络在图像分类任务上相比可达到SOTA；
- 分析了训练时间可重参数化分支和正则化动态松弛在训练中的作用；
- 模型泛化能力更优，性能更好


## 2. 模型性能数据

以下表格是在 RDK X5 & RDK X5 Module 上实际测试得到的性能数据，可以根据自己推理实际需要的性能和精度，对模型的大小做权衡取舍。


| 模型           | 尺寸(像素)  | 类别数  | 参数量(M) | 浮点Top-1  | 量化Top-1  | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) | 帧率      |
| ------------ | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------- |
| MobileOne_S4 | 224x224 | 1000 | 14.8   | 78.75 | 76.50 | 4.58        | 15.44       | 256.52  |
| MobileOne_S3 | 224x224 | 1000 | 10.1   | 77.27 | 75.75 | 2.93        | 9.04        | 437.85  |
| MobileOne_S2 | 224x224 | 1000 | 7.8    | 74.75 | 71.25 | 2.11        | 6.04        | 653.68  |
| MobileOne_S1 | 224x224 | 1000 | 4.8    | 72.31 | 70.45 | 1.31        | 3.69        | 1066.95 |
| MobileOne_S0 | 224x224 | 1000 | 2.1    | 69.25 | 67.58 | 0.80        | 1.59        | 2453.17 |


说明: 
1. X5的状态为最佳状态：CPU为8xA55@1.8G, 全核心Performance调度, BPU为1xBayes-e@1G, 共10TOPS等效int8算力。
2. 单线程延迟为单帧，单线程，单BPU核心的延迟，BPU推理一个任务最理想的情况。
3. 4线程工程帧率为4个线程同时向双核心BPU塞任务，一般工程中4个线程可以控制单帧延迟较小，同时吃满所有BPU到100%，在吞吐量(FPS)和帧延迟间得到一个较好的平衡。
4. 8线程极限帧率为8个线程同时向X3的双核心BPU塞任务，目的是为了测试BPU的极限性能，一般来说4核心已经占满，如果8线程比4线程还要好很多，说明模型结构需要提高"计算/访存"比，或者编译时选择优化DDR带宽。
5. 浮点/定点Top-1：浮点Top-1使用的是模型未量化前onnx的 Top-1 推理精度，量化Top-1则为量化后模型实际推理的精度。

## 3. 模型下载

**.bin 文件下载**：

可以使用脚本 [download.sh](./model/download.sh) 一键下载所有此模型结构的 .bin 模型文件，方便直接更换模型。或者使用以下命令行中的一个，选取单个模型进行下载：

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/MobileOne_S0_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/MobileOne_S1_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/MobileOne_S2_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/MobileOne_S3_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/MobileOne_S4_224x224_nv12.bin
```

**ONNX文件下载**：

onnx 模型使用的是 MobileOne 模型源码进行转换的，使用以下命令安装所需要的包：

```shell
pip install timm onnx
```

模型源码下载使用以下命令：

```shell
git clone https://github.com/apple/ml-mobileone.git
```

模型转换以 mobileone_s0 为例，其余四个模型同理：

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify

from mobileone import *

def count_parameters(onnx_model_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    # Get the initializers (weights in the model)
    initializer = model.graph.initializer
    
    # Calculate the total number of parameters
    total_params = 0
    for tensor in initializer:
        # Get the dimensions of each weight
        dims = tensor.dims
        # Calculate the number of parameters in this weight (product of all dimensions)
        params = 1
        for dim in dims:
            params *= dim
        total_params += params
    
    return total_params

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "mobileone_s0_unfused.pth.tar"
    model = mobileone(variant='s0')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = reparameterize_model(model)

    # print(model)

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "mobileone_s0.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        verbose=True,
        input_names=["data"],  # 输入名
        output_names=["output"],  # 输出名
        keep_initializers_as_inputs=True
    )
    
    # Simplify the ONNX model
    model_simp, check = simplify(onnx_file_path)

    if check:
        print("Simplified model is valid.")
        simplified_onnx_file_path = "mobileone_s0.onnx"
        onnx.save(model_simp, simplified_onnx_file_path)
        print(f"Simplified model saved to {simplified_onnx_file_path}")
    else:
        print("Simplified model is invalid!")
    
    onnx_model_path = simplified_onnx_file_path  # Replace with your ONNX model path
    total_params = count_parameters(onnx_model_path)
    print(f"Total number of parameters in the model: {total_params}")
```

## 4. 部署测试

在下载完毕 .bin 文件后，可以执行 test_MobileOne_*.ipynb 系列的 MobileOne 模型 jupyter 脚本文件，在板端实际运行体验实际测试效果。需要更改测试图片，可额外下载数据集后，放入到data文件夹下并更改 jupyter 文件中图片的路径

![](./data/inference.png)

## 5. 量化实验

若想要进一步进阶对模型量化过程中的学习，如选取量化精度、对模型节点进行取舍、模型输入输出格式配置等，可以按顺序在天工开物工具链（注意是在pc端，不是板端）中执行 mapper 文件夹下的shell文件，对模型进行量化调优。这里仅仅给出 yaml 的配置文件（在yaml文件夹中），如需进行量化实验可将对应不同大小模型的yaml文件自行替换
