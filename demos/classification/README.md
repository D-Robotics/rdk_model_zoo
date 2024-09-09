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

The following table shows the performance data obtained from actual testing on RDK X5 & RDK X5 Module. You can weigh the size of the model according to your own reasoning about the actual performance and accuracy required

| Architecture   | Model       | Size    | Categories | Parameter | Floating point Top-1 | Quantization Top-1 | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ----------- | -------------------------- | ----------- | -------- | --------- | --------- | --------- | ----------- | ----------- | ----------- |
| Transformer | EdgeNeXt_base              | 224x224     | 1000     | 18.51     | 78.21     | 74.52     | 8.80        | 32.31       | 113.35      |
|             | EdgeNeXt_small             | 224x224     | 1000     | 5.59      | 76.50     | 71.75     | 4.41        | 14.93       | 226.15      |
|             | **EdgeNeXt_x_small**       | **224x224** | **1000** | **2.34**  | **71.75** | **66.25** | **2.88**    | **9.63**    | **345.73**  |
|             | EdgeNeXt_xx_small          | 224x224     | 1000     | 1.33      | 69.50     | 64.25     | 2.47        | 7.24        | 403.49      |
|             | EfficientFormer_l3         | 224x224     | 1000     | 31.3      | 76.75     | 76.05     | 17.55       | 65.56       | 60.52       |
|             | **EfficientFormer_l1**     | **224x224** | **1000** | **12.3**  | **76.12** | **65.38** | **5.88**    | **20.69**   | **191.605** |
|             | EfficientFormerv2_s2       | 224x224     | 1000     | 12.6      | 77.50     | 70.75     | 6.99        | 26.01       | 152.40      |
|             | **EfficientFormerv2_s1**   | **224x224** | **1000** | **6.12**  | **77.25** | **68.75** | **4.24**    | **14.35**   | **275.95**  |
|             | EfficientFormerv2_s0       | 224x224     | 1000     | 3.57      | 74.25     | 68.50     | 5.79        | 19.96       | 198.45      |
|             | **EfficientViT_MSRA_m5**   | **224x224** | **1000** | **12.41** | **73.75** | **72.50** | **6.34**    | **22.69**   | **174.70**  |
|             | FastViT_SA12               | 224x224     | 1000     | 10.93     | 78.25     | 74.50     | 11.56       | 42.45       | 93.44       |
|             | FastViT_S12                | 224x224     | 1000     | 8.86      | 76.50     | 72.0      | 5.86        | 20.45       | 193.87      |
|             | **FastViT_T12**            | **224x224** | **1000** | **6.82**  | **74.75** | **70.43** | **4.97**    | **16.87**   | **234.78**  |
|             | FastViT_T8                 | 224x224     | 1000     | 3.67      | 73.50     | 68.50     | 2.09        | 5.93        | 667.21      |
| CNN         | ConvNeXt_nano              | 224x224     | 1000     | 15.59     | 77.37     | 71.75     | 5.71        | 19.80       | 200.18      |
|             | ConvNeXt_pico              | 224x224     | 1000     | 9.04      | 77.25     | 71.03     | 3.37        | 10.88       | 364.07      |
|             | **ConvNeXt_femto**         | **224x224** | **1000** | **5.22**  | **73.75** | **72.25** | **2.46**    | **7.11**    | **556.02**  |
|             | ConvNeXt_atto              | 224x224     | 1000     | 3.69      | 73.25     | 69.75     | 1.96        | 5.39        | 732.10      |
|             | Efficientnet_B4            | 224x224     | 1000     | 19.27     | 74.25     | 71.75     | 5.44        | 18.63       | 212.75      |
|             | Efficientnet_B3            | 224x224     | 1000     | 12.19     | 76.22     | 74.05     | 3.96        | 12.76       | 310.30      |
|             | Efficientnet_B2            | 224x224     | 1000     | 9.07      | 76.50     | 73.25     | 3.31        | 10.51       | 376.77      |
|             | FasterNet_S                | 224x224     | 1000     | 31.18     | 77.04     | 76.15     | 6.73        | 24.34       | 162.83      |
|             | FasterNet_T2               | 224x224     | 1000     | 15.04     | 76.50     | 76.05     | 3.39        | 11.56       | 342.48      |
|             | **FasterNet_T1**           | **224x224** | **1000** | **7.65**  | **74.29** | **71.25** | **1.96**    | **5.58**    | **708.40**  |
|             | FasterNet_T0               | 224x224     | 1000     | 3.96      | 71.75     | 68.50     | 1.41        | 3.48        | 1135.13     |
|             | GoogLeNet                  | 224x224     | 1000     | 6.81      | 68.72     | 67.71     | 2.19        | 6.30        | 626.27      |
|             | MobileNetv1                | 224x224     | 1000     | 1.33      | 71.74     | 65.36     | 1.27        | 2.90        | 1356.25     |
|             | **Mobilenetv2**            | **224x224** | **1000** | **3.44**  | **72.0**  | **68.17** | **1.42**    | **3.43**    | **1152.07** |
|             | **Mobilenetv3_large_100**  | **224x224** | **1000** | **5.47**  | **74.75** | **64.75** | **2.02**    | **5.53**    | **714.22**  |
|             | Mobilenetv4_conv_medium    | 224x224     | 1000     | 9.68      | 76.75     | 75.14     | 2.42        | 6.91        | 572.36      |
|             | **Mobilenetv4_conv_small** | **224x224** | **1000** | **3.76**  | **70.75** | **68.75** | **1.18**    | **2.74**    | **1436.22** |
|             | MobileOne_S4               | 224x224     | 1000     | 14.82     | 78.75     | 76.50     | 4.58        | 15.44       | 256.52      |
|             | MobileOne_S3               | 224x224     | 1000     | 10.19     | 77.27     | 75.75     | 2.93        | 9.04        | 437.85      |
|             | MobileOne_S2               | 224x224     | 1000     | 7.87      | 74.75     | 71.25     | 2.11        | 6.04        | 653.68      |
|             | **MobileOne_S1**           | **224x224** | **1000** | **4.83**  | **72.31** | **70.45** | **1.31**    | **3.69**    | **1066.95** |
|             | **MobileOne_S0**           | **224x224** | **1000** | **2.15**  | **69.25** | **67.58** | **0.80**    | **1.59**    | **2453.17** |
|             | RepGhostNet_200            | 224x224     | 1000     | 9.79      | 76.43     | 75.25     | 2.89        | 8.76        | 451.42      |
|             | RepGhostNet_150            | 224x224     | 1000     | 6.57      | 74.75     | 73.50     | 2.20        | 6.30        | 626.60      |
|             | RepGhostNet_130            | 224x224     | 1000     | 5.48      | 75.00     | 73.57     | 1.87        | 5.30        | 743.56      |
|             | RepGhostNet_111            | 224x224     | 1000     | 4.54      | 72.75     | 71.25     | 1.71        | 4.47        | 881.19      |
|             | **RepGhostNet_100**        | **224x224** | **1000** | **4.07**  | **72.50** | **72.25** | **1.55**    | **4.08**    | **964.69**  |
|             | RepVGG_B1g2                | 224x224     | 1000     | 41.36     | 77.78     | 68.25     | 9.77        | 36.19       | 109.61      |
|             | RepVGG_B1g4                | 224x224     | 1000     | 36.12     | 77.58     | 62.75     | 7.58        | 27.47       | 144.39      |
|             | RepVGG_B0                  | 224x224     | 1000     | 14.33     | 75.14     | 60.36     | 3.07        | 9.65        | 410.55      |
|             | RepVGG_A2                  | 224x224     | 1000     | 25.49     | 76.48     | 62.97     | 6.07        | 21.31       | 186.04      |
|             | **RepVGG_A1**              | **224x224** | **1000** | **12.78** | **74.46** | **62.78** | **2.67**    | **8.21**    | **482.20**  |
|             | RepVGG_A0                  | 224x224     | 1000     | 8.30      | 72.41     | 51.75     | 1.85        | 5.21        | 757.73      |
|             | RepViT_m1_1                | 224x224     | 1000     | 8.27      | 77.73     | 77.50     | 2.32        | 6.69        | 590.42      |
|             | **RepViT_m1_0**            | **224x224** | **1000** | **6.83**  | **76.75** | **76.50** | **1.97**    | **5.71**    | **692.29**  |
|             | RepViT_m0_9                | 224x224     | 1000     | 5.14      | 76.32     | 75.75     | 1.65        | 4.37        | 902.69      |
|             | ResNet18                   | 224x224     | 1000     | 11.27     | 71.49     | 70.50     | 2.95        | 8.81        | 448.79      |
|             | ResNeXt50_32x4d            | 224x224     | 1000     | 24.99     | 76.25     | 76.00     | 5.89        | 20.90       | 189.61      |
|             | VargConvNet                | 224x224     | 1000     | 2.03      | 74.51     | 73.69     | 3.99        | 12.75       | 310.29      |


Description:
1. X5 is in the best state: CPU is 8xA55@1.8G, full core Performance scheduling, BPU is 1xBayes-e@1G, a total of 10TOPS equivalent int8 computing power.
2. Single-threaded delay is the ideal situation for single frame, single-threaded, and single-BPU core delay, and BPU inference for a task.
3. The frame rate of a 4-thread project is when 4 threads simultaneously send tasks to a dual-core BPU. In a typical project, 4 threads can control the single frame delay to be small, while consuming all BPUs to 100%, achieving a good balance between throughput (FPS) and frame delay.
4. The maximum frame rate of 8 threads is for 8 threads to simultaneously load tasks into the dual-core BPU of X3. The purpose is to test the maximum performance of the BPU. Generally, 4 cores are already full. If 8 threads are much better than 4 threads, it indicates that the model structure needs to improve the "calculation/memory access" ratio or optimize the DDR bandwidth when compiling.
5. Floating-point/fixed-point Top-1: Floating-point accuracy uses the Top-1 inference accuracy Level of onnx before the model is quantized, while quantized accuracy is the accuracy Level of the actual inference of the model after quantization.
6. The bold parts in the table are the models recommended for balancing inference speed and accuracy, and models with higher inference accuracy or faster inference speed can be used according to the actual deployment situation

## 3. Model Download Link

The model conversion file for the classification model has been uploaded to Cloud as a Service and can be downloaded from the server website using the wget command.：

Server website address: https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/

Download example:

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EdgeNeXt_x_small_224x224_nv12.bin
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