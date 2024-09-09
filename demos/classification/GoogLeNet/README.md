English | [简体中文](./README_cn.md)

# CNN - GoogLeNet

- [CNN - GoogLeNet](#cnn---googlenet)
  - [1. Introduction](#1-introduction)
  - [2. Model performance data](#2-model-performance-data)
  - [3. Model download](#3-model-download)
  - [4. Deployment Testing](#4-deployment-testing)


## 1. Introduction

- **Paper**: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

- **GitHub repository**: [vision/torchvision/models/googlenet.py at main · pytorch/vision (github.com)](https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py)

GoogLeNet is a deep neural networks model based on the Inception module launched by Google. It won the championship in the ImageNet competition in 2014 and has been improving in the following two years, forming versions such as Inception V2, Inception V3, and Inception V4

Generally speaking, the most direct way to improve network performance is to increase network depth and width, but blindly increasing will bring many problems:

1. There are too many parameters. If the training dataset is limited, it is easy to generate overfitting.
2. The larger the network and the more parameters, the greater the computational complexity, making it difficult to apply.
3. The deeper the network, the easier it is to have layer dispersion problems (the further the layer is traversed, the easier it is to disappear), making it difficult to optimize the model.

We hope to reduce parameters while increasing the depth and width of the network. In order to reduce parameters, we naturally thought of turning full connections into sparse connections. However, in terms of implementation, the actual calculation amount will not be significantly improved after full connections become sparse connections, because most hardware is optimized for dense matrix calculation. Although sparse matrices have less data, it is difficult to reduce the time consumed by calculation. In this demand and situation, Google researchers proposed the Inception method.

![](./data/GoogLeNet_architecture.png)

**GoogLeNet model features**:

- **Modularization Design**: GoogleNet introduced an innovative structure called Inception Module , which extracts multi-scale features by using convolution kernels of different sizes (1x1, 3x3, 5x5) and pooling layers in parallel at each level. This Modularization design effectively reduces the number of parameters and improves computational efficiency.
- **Small number of parameters**: GoogleNet has about 6.8 million parameters, which is significantly smaller than similar deep networks (such as VGG16). This gives it advantages in memory usage and inference speed.
- **Depth**: Despite the small number of parameters, GoogleNet's network is still deep, with a total of 22 layers (excluding pooling layers).
- **Efficient comput**ation: By using 1x1 convolution for dimensionality reduction, GoogleNet reduces computational costs while maintaining high model performance.

## 2. Model performance data

The following table shows the performance data obtained from actual testing on RDK X5 & RDK X5 Module. 

| Model       | Size    | Categories | Parameter | Floating point precision | Quantization accuracy | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ----------- | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------ |
| GoogLeNet | 224x224     | 1000     | 6.81      | 68.72     | 67.71     | 2.19        | 6.30        | 626.27      |

Description:
1. X5 is in the best state: CPU is 8xA55@1.8G, full core Performance scheduling, BPU is 1xBayes-e@1G, a total of 10TOPS equivalent int8 computing power.
2. Single-threaded delay is the ideal situation for single frame, single-threaded, and single-BPU core delay, and BPU inference for a task.
3. The frame rate of a 4-thread project is when 4 threads simultaneously send tasks to a dual-core BPU. In a typical project, 4 threads can control the single frame delay to be small, while consuming all BPUs to 100%, achieving a good balance between throughput (FPS) and frame delay.
4. The maximum frame rate of 8 threads is for 8 threads to simultaneously load tasks into the dual-core BPU of X3. The purpose is to test the maximum performance of the BPU. Generally, 4 cores are already full. If 8 threads are much better than 4 threads, it indicates that the model structure needs to improve the "calculation/memory access" ratio or optimize the DDR bandwidth when compiling.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference accuracy Level of onnx before the model is quantized, while quantized accuracy is the accuracy Level of the actual inference of the model after quantization.


## 3. Model download

**.Bin file download** :

Go into the model folder and use the following command line to download the GoogLeNet model:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/googlenet_224x224_nv12.bin
```

Due to the fact that this model is the output obtained by model quantization by the Horizon Reference algorithm, the model does not provide onnx format files. If you need GoogLeNet model quantization conversion, you can refer to the conversion steps of other models in this repository.

## 4. Deployment Testing

After downloading the .bin file, you can execute the GoogLeNet model jupyter script file of the test_GoogLeNet.ipynb to experience the actual test effect on the board. If you need to change the test picture, you can download the dataset separately and put it in the data folder and change the path of the picture in the jupyter file

![](./data/inference.png)

