English | [简体中文](./README_cn.md)

# CNN X5 - ResNeXt

- [CNN X5 - ResNeXt](#cnn-x5---resnext)
  - [1. Introduction](#1-introduction)
  - [2. Model performance data](#2-model-performance-data)
  - [3. Model download](#3-model-download)
  - [4. Deployment Testing](#4-deployment-testing)
  - [5. Model Quantitation Experiment](#5-model-quantitation-experiment)

## 1. Introduction

- **Paper**: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

- **GitHub repository**: [facebookresearch/ResNeXt: Implementation of a classification framework from the paper Aggregated Residual Transformations for Deep Neural Networks (github.com)](https://github.com/facebookresearch/ResNeXt)

ResNeXt proposes a simple architecture that uses the strategy of VGG/ResNets repeating the same network layer to extend the Split-Transform-Merge strategy in a simple and scalable way. The high-dimensional feature maps in ResNet are grouped into multiple identical low-dimensional feature maps, and then after the convolution operation, the multiple sets of structures are summed to obtain the ResNeXt model.

The key feature of ResNeXt is the introduction of the concept of "cardinality". The ResNeXt module can be seen as a combination of multiple independent paths (i.e. multiple parallel convolution channels). Compared to ResNet, which directly increases the depth and width of the network, ResNeXt improves the expressiveness of the model by increasing the number of these parallel paths

![](./data/ResNet&ResNeXt.png)

**ResNeXt model features**:

- **Cardinality**: This is the core innovation of ResNeXt. The number of cards refers to the number of parallel paths. Increasing the number of cards can improve the performance of the network without significantly increasing the parameters and computation.
- **Grouped Convolution**: ResNeXt uses group convolution, and the convolution operations in multiple groups are independent of each other, which makes the network structure achieve a better balance between computational efficiency and accuracy.
- **Modularization design**: ResNeXt's module design is very concise, each module consists of multiple parallel paths, which share the same hyperparameters, such as convolution kernel size, stride, etc.


## 2. Model performance data

The following table shows the performance data obtained from actual testing on RDK X5 & RDK X5 Module. 

| Model       | Size    | Categories | Parameter | Floating point precision | Quantization accuracy | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ----------- | ------- | ---------- | --------- | ------------------------ | --------------------- | ------------------------------------ | ----------------------------------- | --------------- |
| ResNeXt50_32x4d  | 224x224 | 1000 | 24.99  | 76.25 | 76.00 | 5.89   | 20.90       | 189.61 |

Description:
1. X5 is in the best state: CPU is 8xA55@1.8G, full core Performance scheduling, BPU is 1xBayes-e@1G, a total of 10TOPS equivalent int8 computing power.
2. Single-threaded delay is the ideal situation for single frame, single-threaded, and single-BPU core delay, and BPU inference for a task.
3. The frame rate of a 4-thread project is when 4 threads simultaneously send tasks to a dual-core BPU. In a typical project, 4 threads can control the single frame delay to be small, while consuming all BPUs to 100%, achieving a good balance between throughput (FPS) and frame delay.
4. The maximum frame rate of 8 threads is for 8 threads to simultaneously load tasks into the dual-core BPU of X3. The purpose is to test the maximum performance of the BPU. Generally, 4 cores are already full. If 8 threads are much better than 4 threads, it indicates that the model structure needs to improve the "calculation/memory access" ratio or optimize the DDR bandwidth when compiling.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference accuracy Level of onnx before the model is quantized, while quantized accuracy is the accuracy Level of the actual inference of the model after quantization.


## 3. Model download

**.Bin file download** :

Go into the model folder and use the following command line to download the ResNeXt50_32x4d model:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ResNeXt50_32x4d_224x224_nv12.bin
```

Due to the fact that this model is the output obtained by model quantization by the Horizon Reference algorithm, the model does not provide onnx format files. If you need ResNeXt model quantization conversion, you can refer to the conversion steps of other models in this repository.

## 4. Deployment Testing

After downloading the .bin file, you can execute the ResNeXt50_32x4d model jupyter script file of the test_ResNeXt50_32x4d.ipynb to experience the actual test effect on the board. If you need to change the test picture, you can download the dataset separately and put it in the data folder and change the path of the picture in the jupyter file

![](./data/inference.png)

## 5. Model Quantitation Experiment

If you want to further advance the learning of model quantization, such as selecting quantization accuracy, selecting model nodes, configuring model input and output formats, etc., you can execute the shell file under the mapper folder in the Tiangong Kaiwu toolchain (note that it is on the PC side, not the board side) in order to optimize the model quantization. Here only gives the yaml configuration file (in the yaml folder), if you need to carry out quantization experiments, you can replace the yaml file corresponding to different sizes of models