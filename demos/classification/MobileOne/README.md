English | [简体中文](./README_cn.md)

# CNN X5 - MobileOne

- [CNN X5 - MobileOne](#cnn-x5---mobileone)
  - [1. Introduction](#1-introduction)
  - [2. Model performance data](#2-model-performance-data)
  - [3. Model download](#3-model-download)
  - [4. Deployment Testing](#4-deployment-testing)
  - [5. Model Quantitation Experiment](#5-model-quantitation-experiment)


## 1. Introduction

- **Paper**: [MobileOne: An Improved One millisecond Mobile Backbone](http://arxiv.org/abs/2206.04040)

- **GitHub repository**: [apple/ml-mobileone: This repository contains the official implementation of the research paper, "An Improved One millisecond Mobile Backbone". (github.com)](https://github.com/apple/ml-mobileone)

![](./data/MobileOne_architecture.png)

MobileOne is an efficient visual backbone architecture on end-side devices that utilizes structural re-parameterization technology (iPhone 12, MobileOne's inference time is only 1 millisecond). Moreover, compared with existing architectures deployed on mobile devices, it adopts a **structural re-parameterization** method and does not add commonly used residual connections to speed up inference. MobileOne can be extended to multiple tasks: image classification, object detection, and semantic segmentation, with significant improvements in latency and accuracy. The core module of MobileOne is designed based on MobileNetV1 and absorbs the idea of re-parameterization. The basic architecture used is 3x3 depthwise convolution + 1x1 pointwise convolution


**MobileOne model features**:

- It can run in 1ms on mobile devices (iPhone 12) and achieve SOTA in image classification tasks compared to other efficient/lightweight networks.
- The role of reparameterized branching and regularized dynamic relaxation in training time is analyzed.
- Model Generalization Ability and Performance


## 2. Model performance data

The following table shows the performance data obtained from actual testing on RDK X5 & RDK X5 Module. You can weigh the size of the model according to your own reasoning about the actual performance and accuracy required


| Model        | Size    | Categories | Parameter | Floating point precision | Quantization accuracy | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ------------ | ------- | ---------- | --------- | ------------------------ | --------------------- | ------------------------------------ | ----------------------------------- | --------------- |
| MobileOne_S4 | 224x224 | 1000       | 14.8      | 78.75                    | 76.50                 | 4.58                                 | 15.44                               | 256.52          |
| MobileOne_S3 | 224x224 | 1000       | 10.1      | 77.27                    | 75.75                 | 2.93                                 | 9.04                                | 437.85          |
| MobileOne_S2 | 224x224 | 1000       | 7.8       | 74.75                    | 71.25                 | 2.11                                 | 6.04                                | 653.68          |
| MobileOne_S1 | 224x224 | 1000       | 4.8       | 72.31                    | 70.45                 | 1.31                                 | 3.69                                | 1066.95         |
| MobileOne_S0 | 224x224 | 1000       | 2.1       | 69.25                    | 67.58                 | 0.80                                 | 1.59                                | 2453.17         |


Description:
1. X5 is in the best state: CPU is 8xA55@1.8G, full core Performance scheduling, BPU is 1xBayes-e@1G, a total of 10TOPS equivalent int8 computing power.
2. Single-threaded delay is the ideal situation for single frame, single-threaded, and single-BPU core delay, and BPU inference for a task.
3. The frame rate of a 4-thread project is when 4 threads simultaneously send tasks to a dual-core BPU. In a typical project, 4 threads can control the single frame delay to be small, while consuming all BPUs to 100%, achieving a good balance between throughput (FPS) and frame delay.
4. The maximum frame rate of 8 threads is for 8 threads to simultaneously load tasks into the dual-core BPU of X3. The purpose is to test the maximum performance of the BPU. Generally, 4 cores are already full. If 8 threads are much better than 4 threads, it indicates that the model structure needs to improve the "calculation/memory access" ratio or optimize the DDR bandwidth when compiling.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference Confidence Level of onnx before the model is quantized, while quantized accuracy is the Confidence Level of the actual inference of the model after quantization.


## 3. Model download

**.Bin file download** :

You can use the script [download_bin.sh](./model/download_bin.sh) to download all .bin model files for this model structure with one click, making it easy to change models directly. Alternatively, use one of the following command lines to select a single model for download:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/MobileOne_S0-deploy_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/MobileOne_S1-deploy_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/MobileOne_S2-deploy_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/MobileOne_S3-deploy_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/MobileOne_S4-deploy_224x224_nv12.bin
```

**ONNX file download** :

Similarly to the .bin file, use [download_onnx.sh](./model/download_onnx.sh) to download all .onnx model files of this model structure with one click, or download a single .onnx model for quantization experiments:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/mobileone_s0_deploy.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/mobileone_s1_deploy.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/mobileone_s2_deploy.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/mobileone_s3_deploy.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/mobileone_s4_deploy.onnx
```

## 4. Deployment Testing

After downloading the .bin file, you can execute the EdgeNeXt model jupyter script file of the test_EdgeNeXt_ * .ipynb series to experience the actual test effect on the board. If you need to change the test picture, you can download the dataset separately and put it in the data folder and change the path of the picture in the jupyter file

![](./data/inference.png)

## 5. Model Quantitation Experiment

If you want to further advance the learning of model quantization, such as selecting quantization accuracy, selecting model nodes, configuring model input and output formats, etc., you can execute the shell file under the mapper folder in the Tiangong Kaiwu toolchain (note that it is on the PC side, not the board side) in order to optimize the model quantization. Here only gives the yaml configuration file (in the yaml folder), if you need to carry out quantization experiments, you can replace the yaml file corresponding to different sizes of models