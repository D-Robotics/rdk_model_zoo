English | [简体中文](./README_cn.md)

# Classification

- [Classification](#classification)
  - [1. Introduction to Model Classification](#1-introduction-to-model-classification)
  - [2. Model Performance Data](#2-model-performance-data)
  - [3. Model Download Link](#3-model-download-link)
  - [4. Input/Output Data](#4-inputoutput-data)
  - [5. PTQ Quantification Process](#5-ptq-quantification-process)

## 1. Introduction to Model Classification

Model classification refers to the process of **classifying input data** by a Machine Learning model. Specifically, the task of a classification model is to assign input data (such as images, text, audio, etc.) to predefined categories. These categories are usually discrete labels, such as classifying images as "cats" or "dogs", or classifying emails as "junk email" or "non-junk email".

In Machine Learning, the training process of classification models includes the following step：

- **Data Acquisition**: This process includes image collection and annotation of corresponding tags.
- **Data preprocessing**: Cleaning and transforming data to fit the input requirements of the model (such as image size cropping and scaling).
- **Feature extraction**: extracting useful features from the original data source, usually to convert the data into a format suitable for model processing (such as channel sorting for converting models from bgr to rgb, NCHW to NHWC).
- **Model Training**: Use labeled data to train classification models, adjust Model Training parameters to improve model detection accuracy and confidence level.
- **Model validation**: Use validation set data to evaluate the performance of the model and fine-tune and optimize the model.
- **Model testing**: Evaluate the final performance of the model on the test set to determine its generalization ability.
- **Deployment and application**: the trained model for model quantization and edge side plate end deployment.

Deep learning classification models are generally applied to classify images according to their labels. The most famous image classification challenge is ImageNet Classification. ImageNet Classification is an important image classification challenge in the field of Deep learning. The dataset was built by Professor Li Feifei and her team at Stanford University in 2007 and published at CVPR in 2009. The ImageNet dataset is widely used for image classification, detection, and localization tasks.

ILSVRC (ImageNet Large-Scale Visual Recognition Challenge) is a competition based on the ImageNet dataset. It was first held in 2010 and ended in 2017. The competition includes image classification, object localization, object detection, video object detection, and scene classification. Previous winners include famous deep learning models such as AlexNet, VGG, GoogLeNet, and ResNet. The ILSVRC2012 dataset is the most commonly used subset, containing 1000 classifications, each with about 1000 images, totaling about 1.20 million training images, as well as 50,000 validation images and 100,000 test images (the test set has no labels).

ImageNet official download address：https://image-net.org/

![](../../resource/imgs/ImageNet.png)

Due to the fact that the models provided by this repository are all onnx/bin files obtained after pre-training models are transformed, there is no need to perform Model Training operations again (lack of training resources is also a big problem). Due to the large dataset, the ImageNet dataset is used as a calibration dataset for subsequent model quantization operations. The following table shows the size of the ImageNet ILSVRC2012 dataset.


| Dataset Type | Category | Number of Images |
| -------------- | ---- | ------- |
| ILSVRC2012 training dataset | 1000 | 1.20 million images |
| ILSVRC2012 validation set | 1000 | 50,000 images |
| ILSVRC2012 test set | 1000 | 100,000 images |

ILSVRC2012 is a subset of ImageNet, which itself has over 14 million images with over 20,000 categories. Over 1 million of these images have explicit category labels and object position labels.

For the evaluation of image recognition results based on ImageNet, two accuracy indicators are often used, one is top-1 accuracy and the other is top-5 accuracy. **Top-1 accuracy refers to the probability that the largest output probability corresponds to the correct category; top-5 accuracy refers to the probability that the five largest output probabilities correspond to the five categories that contain the correct category**. This repository provides Top-5 accuracy category predictions, which can more clearly compare the output results of the model.

## 2. Model Performance Data

The following table shows the performance data obtained from actual testing on RDK X3 & RDK X3 Module. 

| Architecture   | Model       | Size    | Categories | Parameter | Floating point Top-1 | Quantization Top-1 | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ----------- | -------------------------- | ----------- | -------- | --------- | --------- | --------- | ----------- | ----------- | ----------- |
| CNN | GoogLeNet                  | 224x224     | 1000     | 6.81      | 68.72     | 67.71     | 8.34        | 16.29       | 243.51      |
|     | Mobilenetv4                | 224x224     | 1000     | 3.76      | 70.50     | 70.26     | 1.43        | 2.96        | 1309.17     |
|     | Mobilenetv2                | 224x224     | 1000     | 3.4       | 72.0      | 68.17     | 2.41        | 4.42        | 890.99      |
|     | Mobilenetv4                | 224x224     | 1000     | 3.76      | 70.50     | 70.26     | 1.43        | 2.96        | 1309.17     |
|     | MobileOne                  | 224x224     | 1000     | 4.8       | 72.00     | 71.00     | 4.50        | 8.70        | 455.87      |
|     | RepGhost                   | 224x224     | 1000     | 4.07      | 72.50     | 72.25     | 2.09        | 4.56        | 855.18      |
|     | RepVGG                     | 224x224     | 1000     | 12.78     | 74.46     | 62.78     | 11.58       | 22.71       | 174.94      |
|     | RepViT                     | 224x224     | 1000     | 5.1       | 75.25     | 75.75     | 28.34       | 41.22       | 96.47       |
|     | ResNet18                   | 224x224     | 1000     | 11.2      | 71.49     | 70.50     | 8.87        | 17.07       | 232.74      |


Description:
1. X3 is in the best state: CPU is 4xA53@1.5G, full core Performance scheduling, BPU is 2xBernoulli@1G, a total of 5TOPS equivalent int8 computing power.
2. Single-threaded delay is the ideal situation for single frame, single-threaded, and single-BPU core delay, and BPU inference for a task.
3. The frame rate of a 4-thread project is when 4 threads simultaneously send tasks to a dual-core BPU. In a typical project, 4 threads can control the single frame delay to be small, while consuming all BPUs to 100%, achieving a good balance between throughput (FPS) and frame delay.
4. The maximum frame rate of 8 threads is for 8 threads to simultaneously load tasks into the dual-core BPU of X3. The purpose is to test the maximum performance of the BPU. Generally, 4 cores are already full. If 8 threads are much better than 4 threads, it indicates that the model structure needs to improve the "calculation/memory access" ratio or optimize the DDR bandwidth when compiling.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference accuracy Level of onnx before the model is quantized, while quantized accuracy is the accuracy Level of the actual inference of the model after quantization.
6. The bold parts in the table are the models recommended for balancing inference speed and accuracy, and models with higher inference accuracy or faster inference speed can be used according to the actual deployment situation

## 3. Model Download Link

The model conversion file for the classification model has been uploaded to Cloud as a Service and can be downloaded from the server website using the wget command.：

Server website address: https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x3/

Download example:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x3/RepViT_224x224_nv12.bin
```

Model download shell script In the model folder of each classification model, you can execute the `sh download.sh` command to download the bin file.

## 4. Input/Output Data

- Input Data

| Model Category | Input Data  | Data Type   | Shape  | Layout |
| ---- | ----- | ------- | ----------------- | ------ |
| onnx | image | FLOAT32 | 1 x 3 x 224 x 224 | NCHW   |
| bin  | image | NV12    | 1 x 224 x 224 x 3 | NHWC   |

- Output data


| Model Category | Output Data | Data Type   | shape | Layout |
| ---- | ------- | ------- | -------- | ---- |
| onnx | classes | FLOAT32 | 1 x 1000 | NC   |
| bin  | classes | FLOAT32 | 1 x 1000 | NC   |


The input data shape of the pre-trained pth model is `1x3x224x224` ，and the output data shape is `1x1000`, which meets the format output requirements of ImageNet classification model. Specifically, for the quantization step, the input data format of the bin file of the model quantization is unified as `nv12`'. The format conversion function has been built into the model preprocessing part `bpu_libdnn`, which can be directly called to process the model data.


## 5. PTQ Quantification Process

Model quantization is a technique that **converts floating-point operations into fixed-point operations** of neural networks. It is mainly used to **reduce computational complexity and model size**. It is suitable for resource-constrained embedded devices such as smartphones, drones, and robots. These devices usually have strict requirements for memory, power consumption, and inference time. Model quantization solves these problems by reducing bits.

In regular precision models, FP32 (32-bit floating-point numbers) are usually used for calculation. However, low-precision models (such as INT8, 8-bit fixed-point integers) can significantly reduce model size and speed up inference. Mixed-precision models combine FP32 and FP16 to reduce memory usage, but still retain FP32 in critical computation to ensure accuracy (i.e. mixed-precision quantization).

**Model quantization can significantly reduce model size**. For example, INT8 quantization can reduce model size to one-fourth of its original size. This is particularly important for devices with limited storage space. Secondly, quantized models reduce memory usage and speed up inference. In addition, many hardware accelerators (such as DSP and NPU) only support INT8 operations, further highlighting the necessity of quantization. In the industry, INT8 quantization is widely used, and many mobile end inference frameworks such as NCNN and TNN support the inference function of such models. Typically, these frameworks implement the conversion between FP32 and INT8 by introducing a quantization layer (Quantize) and a dequantize layer (Dequantize), thereby optimizing the performance of the model while ensuring accuracy.

If developers want to further understand the quantization part of the board, the document also provides a reference for [Model Quantization Deployment](./Model%20quantization%20deployment.md), which provides developers with a more detailed quantization step and process method.