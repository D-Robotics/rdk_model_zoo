English | [简体中文](./README_cn.md)

# CNN - ResNet

- [CNN - ResNet](#cnn---resnet)
  - [1. Introduction](#1-introduction)
  - [2. Model performance data](#2-model-performance-data)
  - [3. Model download](#3-model-download)
  - [4. Deployment Testing](#4-deployment-testing)


## 1. Introduction

- **Paper**: [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/2307.09283)

- **GitHub repository**: [vision/resnet.py at main · pytorch/vision (github.com)](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

ResNet was proposed by He Kaiming, Zhang Xiangyu, Ren Shaoqing, Sun Jian and others from Microsoft Research Institute. It won the championship in ILSVRC (ImageNet Large Scale Visual Recognition Challenge) in 2015, with an error rate of 3.57% in the top 5 and lower parameter requirements than VGGNet. Its effect is outstanding and it is a milestone event in the history of convolution neural networks. The author, He Kaiming, also won the CVPR2016 Best Paper Award for this.

The main contribution of "ResNet" is the discovery of the degradation phenomenon (Degradation) , and the use of shortcut connections (Shortcut connections) for the degradation phenomenon, which greatly eliminates the difficulty of training neural networks with excessive depth, and makes the "depth" of neural networks break through 100 layers for the first time, and the largest neural networks even exceed 1000 layers . The structure of "ResNet" can accelerate the training of neural networks extremely quickly , and the accuracy of the model has also been greatly improved. At the same time, the generalization of "ResNet" is very good, and it can even be directly used in "InceptionNet" networks.

The ResNet network refers to the VGG19 network, modifies it, and adds residual units through shortcut connections. The changes are mainly reflected in the direct use of convolution with step size = 2 for downsampling inResNet , and replaces the fully connected layer with a global average pooling layer. An important design principle ofResNet is: When the feature map size is reduced by half, the number of feature maps doubles, which maintains the complexity of the network layer As can be seen from the figure, ResNet adds a short-circuit mechanism between every two layers compared to ordinary networks, which forms residual learning, where the dashed line represents the change in the number of feature maps. The 34-layer ResNet shown in the figure can also build deeper networks as shown in the table. From the table, it can be seen that for the 18-layer and 34-layer ResNet , the residual learning between the two layers is performed. When the network is deeper, the residual learning between the three layers is performed. The three-layer convolution kernels are 1x1 , 3x3 , and 1x1. It is worth noting that the number of feature maps in the hidden layer is relatively small, and it is 1/4 of the output feature map number.

![](./data/ResNet_architecture2.png)
![](./data/ResNet_architecture.png)

**ResNet model features**:

- The residual structure artificially constructs an identity mapping, which can make the entire structure converge in the direction of the identity mapping, ensuring that the final error rate will not increase as the depth increases
- The motivation of ResNet is to solve the degradation problem. The design of residual blocks makes it easy to learn identity mappings. Even if too many blocks are stacked, ResNet can learn redundant blocks into identity mappings without performance degradation.
- The actual depth of the network is determined during the training process, that is, ResNet has a certain ability of deep Self-Adaptation


## 2. Model performance data

The following table shows the performance data obtained from actual testing on RDK X5 & RDK X5 Module. 

| Model       | Size    | Categories | Parameter | Floating point precision | Quantization accuracy | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ----------- | ------- | ---------- | --------- | ------------------------ | --------------------- | ------------------------------------ | ----------------------------------- | --------------- |
| ResNet18 | 224x224 | 1000 | 11.2    | 71.49 | 70.50 | 2.95        | 8.81        | 448.79 |

Description:
1. X5 is in the best state: CPU is 8xA55@1.8G, full core Performance scheduling, BPU is 1xBayes-e@1G, a total of 10TOPS equivalent int8 computing power.
2. Single-threaded delay is the ideal situation for single frame, single-threaded, and single-BPU core delay, and BPU inference for a task.
3. The frame rate of a 4-thread project is when 4 threads simultaneously send tasks to a dual-core BPU. In a typical project, 4 threads can control the single frame delay to be small, while consuming all BPUs to 100%, achieving a good balance between throughput (FPS) and frame delay.
4. The maximum frame rate of 8 threads is for 8 threads to simultaneously load tasks into the dual-core BPU of X3. The purpose is to test the maximum performance of the BPU. Generally, 4 cores are already full. If 8 threads are much better than 4 threads, it indicates that the model structure needs to improve the "calculation/memory access" ratio or optimize the DDR bandwidth when compiling.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference accuracy Level of onnx before the model is quantized, while quantized accuracy is the accuracy Level of the actual inference of the model after quantization.


## 3. Model download

**.Bin file download** :

Go into the model folder and use the following command line to download the ResNet18 model:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ResNet18_224x224_nv12.bin
```

Due to the fact that this model is the output obtained by model quantization by the Horizon Reference algorithm, the model does not provide onnx format files. If you need ResNet model quantization conversion, you can refer to the conversion steps of other models in this repository.

## 4. Deployment Testing

After downloading the .bin file, you can execute main.py to conduct actual operation on the board and test the actual effect. If you need to change the test images, you can download the dataset separately, place it in the data folder, and modify the path of the images in the main.py file.

![](./data/inference.png)

