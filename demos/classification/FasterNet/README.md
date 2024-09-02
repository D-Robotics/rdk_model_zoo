English | [简体中文](./README_cn.md)

# CNN X5 - FasterNet

- [CNN X5 - FasterNet](#cnn-x5---fasternet)
  - [1. Introduction](#1-introduction)
  - [2. Model performance data](#2-model-performance-data)
  - [3. Model download](#3-model-download)
  - [4. Deployment Testing](#4ment-testing)
  - [5. Model Quantitation Experiment](#5-model-quantitation-experiment)


## 1. Introduction

- **Paper**: [Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks](http://arxiv.org/abs/2303.03667)

- **GitHub repository**: [GitHub - JierunChen/FasterNet: [CVPR 2023] Code release for PConv and FasterNet](https://github.com/JierunChen/FasterNet)

The evaluation of other lightweight network performance is based on the number of operations of FLOPs, using group convolution or deep separable convolution as its network components to extract network features, but increasing memory access and computational complexity (concatenation, shuffling, pooling). The ViT structure also requires underlying hardware support. The paper investigated several common network FLOPs and found that they were all lower than ResNet, that is, their computational speed was low

![](./data/FLOPs%20of%20Nets.png)

FasterNet effectively improves spatial features by reducing redundant computation and memory access, and achieves faster running speed than other networks; a FasterNet network skeleton based on partial convolution (PConv) is proposed to make feature extraction more effective

![](./data/FasterNet_architecture.png)


**FasterNet model features**:

- Proposed formulas for evaluating computational latency, pointing out the importance of achieving higher FLOPS, not just reducing FLOPs for faster neural networks
- Two commonly used lightweight network structures, group convolution and deep separable convolution, were used to conduct extensive experiments on various tasks and verify the high speed and effectiveness of PConv and FasterNet


## 2. Model performance data

The following table shows the performance data obtained from actual testing on RDK X5 & RDK X5 Module. You can weigh the size of the model according to your own reasoning about the actual performance and accuracy required


| Model        | Size    | Categories | Parameter | Floating point precision | Quantization accuracy | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ------------ | ------- | ---------- | --------- | ------------------------ | --------------------- | ------------------------------------ | ----------------------------------- | --------------- |
| FasterNet_S  | 224x224 | 1000 | 31.1   | 77.04 | 76.15 | 6.73        | 24.34       | 162.83  |
| FasterNet_T2 | 224x224 | 1000 | 15.0   | 76.50 | 76.05 | 3.39        | 11.56       | 342.48  |
| FasterNet_T1 | 224x224 | 1000 | 7.6    | 74.29 | 71.25 | 1.96        | 5.58        | 708.40  |
| FasterNet_T0 | 224x224 | 1000 | 3.9    | 71.75 | 68.50 | 1.41        | 3.48        | 1135.13 |


Description:
1. X5 is in the best state: CPU is 8xA55@1.8G, full core Performance scheduling, BPU is 1xBayes-e@1G, a total of 10TOPS equivalent int8 computing power.
2. Single-threaded delay is the ideal situation for single frame, single-threaded, and single-BPU core delay, and BPU inference for a task.
3. The frame rate of a 4-thread project is when 4 threads simultaneously send tasks to a dual-core BPU. In a typical project, 4 threads can control the single frame delay to be small, while consuming all BPUs to 100%, achieving a good balance between throughput (FPS) and frame delay.
4. The maximum frame rate of 8 threads is for 8 threads to simultaneously load tasks into the dual-core BPU of X3. The purpose is to test the maximum performance of the BPU. Generally, 4 cores are already full. If 8 threads are much better than 4 threads, it indicates that the model structure needs to improve the "calculation/memory access" ratio or optimize the DDR bandwidth when compiling.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference accuracy Level of onnx before the model is quantized, while quantized accuracy is the accuracy Level of the actual inference of the model after quantization.


## 3. Model download

**.Bin file download** :

You can use the script [download_bin.sh](./model/download_bin.sh) to download all .bin model files for this model structure with one click, making it easy to change models directly. Alternatively, use one of the following command lines to select a single model for download:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/FasterNet_S_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/FasterNet_T0_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/FasterNet_T1_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/FasterNet_T2_224x224_nv12.bin
```

**ONNX file download** :

Similarly to the .bin file, use [download_onnx.sh](./model/download_onnx.sh) to download all .onnx model files of this model structure with one click, or download a single .onnx model for quantization experiments:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/fasternet_s.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/fasternet_t2.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/fasternet_t1.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/fasternet_t0.onnx
```

## 4. Deployment Testing

After downloading the .bin file, you can execute the FasterNet model jupyter script file of the test_FasterNet_ * .ipynb series to experience the actual test effect on the board. If you need to change the test picture, you can download the dataset separately and put it in the data folder and change the path of the picture in the jupyter file

![alt text](./data/inference.png)

## 5. Model Quantitation Experiment

If you want to further advance the learning of model quantization, such as selecting quantization accuracy, selecting model nodes, configuring model input and output formats, etc., you can execute the shell file under the mapper folder in the Tiangong Kaiwu toolchain (note that it is on the PC side, not the board side) in order to optimize the model quantization. Here only gives the yaml configuration file (in the yaml folder), if you need to carry out quantization experiments, you can replace the yaml file corresponding to different sizes of models