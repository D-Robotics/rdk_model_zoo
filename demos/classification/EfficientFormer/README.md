English | [简体中文](./README_cn.md)

# Transformer - EfficientFormer

- [Transformer - EfficientFormer](#transformer---efficientformer)
  - [1. Introduction](#1-introduction)
  - [2. Model performance data](#2-model-performance-data)
  - [3. Model download](#3-model-download)
  - [4. Deployment Testing](#4ment-testing)
  - [5. Model Quantitation Experiment](#5-model-quantitation-experiment)


## 1. Introduction

- **Paper**: [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)

- **GitHub repository**: [EfficientFormer](https://github.com/snap-research/EfficientFormer)

Transformer has slow inference speed and is difficult to deploy on mobile end. Although a lot of work has been done before, this problem has not been well solved. In the figure, the author analyzed the operation of different models on the end side, mainly including Patch Embedding for ViT to block images, Attention and MLP in Transformer, as well as Reshape and some activations proposed by LeViT. Based on the table below, several conjectures were proposed, and then the EfficientFormer structure was designed.

![alt text](./data/latency_profiling.png)

EfficientFormer can run at the speed of MobileNet on mobile devices, identify inefficient operators in a series of ViT architectures, and guide new design paradigms, outperforming lightweight networks such as MobileViT. The EfficientFormer model outperforms existing Transformer models and is faster than the most competitive CNNs. The delay-driven analysis and experimental results of the ViT structure verify the author's statement: **a powerful visual Transformer can achieve ultra-fast inference speed at the edge**.

![alt text](./data/EfficientFormer_architecture.png)

**EfficientFormer model features**:

- Re-explore the design principles of ViT and its variants through delay analysis. Based on existing work, use iPhone 12 as a test platform and public CoreML as a compiler, because mobile devices are widely used, the results can be easily reproduced.
- Secondly, inefficient designs and operations in ViT are identified, and a new dimensional consistency design paradigm is proposed for visual transformers.
- Starting from a supernetwork with a new design paradigm, a simple and effective delay-driven slimming method is proposed to obtain a new model, namely EfficientFormer, which directly optimizes inference speed.


## 2. Model performance data

The following table shows the performance data obtained from actual testing on RDK X5 & RDK X5 Module. You can weigh the size of the model according to your own reasoning about the actual performance and accuracy required

| Model              | Size    | Categories | Parameter | Floating point precision | Quantization accuracy | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ------------------ | ------- | ---------- | --------- | ------------------------ | --------------------- | ------------------------------------ | ----------------------------------- | --------------- |
| EfficientFormer_l3 | 224x224 | 1000 | 31.3   | 76.75 | 76.05 | 17.55       | 65.56       | 60.52   |
| EfficientFormer_l1 | 224x224 | 1000 | 12.3   | 76.75 | 67.72 | 5.88        | 20.69       | 191.605 |

Description:
1. X5 is in the best state: CPU is 8xA55@1.8G, full core Performance scheduling, BPU is 1xBayes-e@1G, a total of 10TOPS equivalent int8 computing power.
2. Single-threaded delay is the ideal situation for single frame, single-threaded, and single-BPU core delay, and BPU inference for a task.
3. The frame rate of a 4-thread project is when 4 threads simultaneously send tasks to a dual-core BPU. In a typical project, 4 threads can control the single frame delay to be small, while consuming all BPUs to 100%, achieving a good balance between throughput (FPS) and frame delay.
4. The maximum frame rate of 8 threads is for 8 threads to simultaneously load tasks into the dual-core BPU of X3. The purpose is to test the maximum performance of the BPU. Generally, 4 cores are already full. If 8 threads are much better than 4 threads, it indicates that the model structure needs to improve the "calculation/memory access" ratio or optimize the DDR bandwidth when compiling.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference Confidence Level of onnx before the model is quantized, while quantized accuracy is the Confidence Level of the actual inference of the model after quantization.


## 3. Model download

**.Bin file download**:

You can use the script [download_bin.sh](./model/download_bin.sh) to download all .bin model files for this model structure with one click, making it easy to change models directly. Alternatively, use one of the following command lines to select a single model for download:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EfficientFormer_l1_224x224_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EfficientFormer_l1_224x224_nv12.bin
```

**ONNX file download**:

Similarly to the .bin file, use [download_onnx.sh](./model/download_onnx.sh) to download all .onnx model files of this model structure with one click, or download a single .onnx model for quantization experiments:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/efficientformer_l1.onnx
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/efficientformer_l3.onnx
```

## 4. Deployment Testing

After downloading the .bin file, you can execute the EfficientFormerV2 model jupyter script file of the test_EfficientFormerV2_ * .ipynb series to experience the actual test effect on the board. If you need to change the test picture, you can download the dataset separately and put it in the data folder and change the path of the picture in the jupyter file.

![](./data/inference.png)


## 5. Model Quantitation Experiment

If you want to further advance the learning of model quantization, such as selecting quantization accuracy, selecting model nodes, configuring model input and output formats, etc., you can execute the shell file under the mapper folder in the Tiangong Kaiwu toolchain (note that it is on the PC side, not the board side) in order to optimize the model quantization.

EfficientFormerV2 due to the internal softmax node, Tiangong Kaiwu toolchain default softmax node on the CPU execution, need to be in the yaml configuration file under the model_parameters parameter node_info softmax quantization in BPU. Here only gives the yaml configuration file (in the yaml folder), if you need to carry out quantization experiments, you can replace the yaml file corresponding to different sizes of models.