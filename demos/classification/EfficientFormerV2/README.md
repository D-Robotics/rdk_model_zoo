English | [简体中文](./README_cn.md)

# Transformer - EfficientFormerV2

- [Transformer - EfficientFormerV2](#transformer---efficientformerv2)
  - [1. Introduction](#1-introduction)
  - [2. Model performance data](#2-model-performance-data)
  - [3. Model download](#3-model-download)
  - [4. Deployment Testing](#4ment-testing)
  - [5. Model Quantitation Experiment](#5-model-quantitation-experiment)

## 1. Introduction

- **Paper**: [EfficientFormerV2: Rethinking Vision Transformers for MobileNet Size and Speed](https://arxiv.org/abs/2212.08059)

- **GitHub repository**: [EfficientFormer](https://github.com/snap-research/EfficientFormer)

![](./data/EfficientFormerV2_architecture.png)

EfficientFormerV2 uses a hybrid visual backbone network and is suitable for mobile devices, proposing a fine grain joint search for size and speed. The model is both lightweight and extremely fast in inference speed. The paper adopts a four-stage hierarchical design, and the obtained feature size is {1/4, 1/8, 1/16, 1/32} of the input resolution. EfficientFormerV2 starts embedding input images from a small kernel convolution stem instead of using inefficient embedding with non-overlapping patches.

EfficientFormerV2 directly removes the original Pooling layer (the larger the downsampling, the larger the theoretical receptive field), and replaces it with the form of BottleNeck. First, it uses a 1x1 convolution to reduce dimensionality and compress, then embeds a 3x3 depth-separable convolution to extract local information, and finally uses a 1x1 convolution to increase dimensionality. One advantage of doing this is that this modification is conducive to directly using hyperparameter search technology to search for the network depth of specific modules in the later stages of the network, so as to extract local and global information.

**EfficientFormerV2 model features**:

- Proposed a new hypernetwork design method that, while maintaining high accuracy, can run on mobile devices
- Proposed a fine grain joint search strategy that simultaneously optimizes latency and number of parameters to find efficient architectures
- Accuracy of EfficientFormerV2 model on ImageNet-1K dataset Approximately 4% higher than MobileNetV2 and MobileNetV2 × 1.4 while having similar latency and parameters

## 2. Model performance data

The following table shows the performance data obtained from actual testing on RDK X5 & RDK X5 Module. You can weigh the size of the model according to your own reasoning about the actual performance and accuracy required


| Model                | Size    | Categories | Parameter | Floating point precision | Quantization accuracy | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| -------------------- | ------- | ---------- | --------- | ------------------------ | --------------------- | ------------------------------------ | ----------------------------------- | --------------- |
| EfficientFormerv2_s2 | 224x224  | 1000  | 12.6   | 77.50  | 70.75  | 6.99        | 26.01       | 152.40 |
| EfficientFormerv2_s1 | 224x224  | 1000  | 6.1    | 77.25  | 68.75  | 4.24        | 14.35       | 275.95 |
| EfficientFormerv2_s0 | 224x224  | 1000  | 3.5    | 74.25  | 68.50  | 5.79        | 19.96       | 198.45 |

Description:
1. X5 is in the best state: CPU is 8xA55@1.8G, full core Performance scheduling, BPU is 1xBayes-e@1G, a total of 10TOPS equivalent int8 computing power.
2. Single-threaded delay is the ideal situation for single frame, single-threaded, and single-BPU core delay, and BPU inference for a task.
3. The frame rate of a 4-thread project is when 4 threads simultaneously send tasks to a dual-core BPU. In a typical project, 4 threads can control the single frame delay to be small, while consuming all BPUs to 100%, achieving a good balance between throughput (FPS) and frame delay.
4. The maximum frame rate of 8 threads is for 8 threads to simultaneously load tasks into the dual-core BPU of X3. The purpose is to test the maximum performance of the BPU. Generally, 4 cores are already full. If 8 threads are much better than 4 threads, it indicates that the model structure needs to improve the "calculation/memory access" ratio or optimize the DDR bandwidth when compiling.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference accuracy Level of onnx before the model is quantized, while quantized accuracy is the accuracy Level of the actual inference of the model after quantization.

## 3. Model download

**.Bin file download**:

You can use the script [download_bin.sh](./model/download_bin.sh) to download all .bin model files for this model structure with one click, making it easy to change models directly. Alternatively, use one of the following command lines to select a single model for download:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EfficientFormerv2_s2_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EfficientFormerv2_s1_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EfficientFormerv2_s0_224x224_nv12.bin
```

**ONNX file download**:

Similarly to the .bin file, use [download_onnx.sh](./model/download_onnx.sh) to download all .onnx model files of this model structure with one click, or download a single .onnx model for quantization experiments:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/efficientformerv2_s2.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/efficientformerv2_s1.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/efficientformerv2_s0.onnx
```

## 4. Deployment Testing

After downloading the .bin file, you can execute the EfficientFormerV2 model jupyter script file of the test_EfficientFormerV2_ * .ipynb series to experience the actual test effect on the board. If you need to change the test picture, you can download the dataset separately and put it in the data folder and change the path of the picture in the jupyter file

![alt text](./data/inference.png)

## 5. Model Quantitation Experiment

If you want to further advance the learning of model quantization, such as selecting quantization accuracy, selecting model nodes, configuring model input and output formats, etc., you can execute the shell file under the mapper folder in the Tiangong Kaiwu toolchain (note that it is on the PC side, not the board side) in order to optimize the model quantization.

EfficientFormerV2 due to the internal softmax node, Tiangong Kaiwu toolchain default softmax node on the CPU execution, need to be in the yaml configuration file under the model_parameters parameter node_info softmax quantization in BPU. Here only gives the yaml configuration file (in the yaml folder), if you need to carry out quantization experiments, you can replace the yaml file corresponding to different sizes of models.