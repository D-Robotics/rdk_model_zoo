[English](./README.md) | 简体中文

# PaddleOCR 文字识别

- [PaddleOCR 文字识别](#paddleocr-文字识别)
  - [1. PaddleOCR 简介](#1-paddleocr-简介)
  - [2. 性能数据](#2-性能数据)
  - [3. 模型下载地址](#3-模型下载地址)
  - [4. 部署测试](#4-部署测试)


## 1. PaddleOCR 简介

PaddleOCR 是百度飞桨基于深度学习的光学字符识别（OCR）工具，利用 PaddlePaddle 框架来执行图片中的文字识别任务。该仓库通过图像预处理、文字检测、文字识别等多个阶段，将图像中的文字转换为可编辑的文本。PaddleOCR 支持多语言和多字体的识别，适合各种复杂场景下的文字提取任务。PaddleOCR 还支持自定义训练，用户可以根据特定需求准备训练数据，进一步优化模型表现。

在实际应用中，PaddleOCR 的工作流程包括以下几个步骤：

- **图像预处理**：对输入的图像进行去噪、尺寸调整等处理，使其适合后续的检测和识别。
- **文字检测**：通过深度学习模型检测图像中的文字区域，生成检测框。
- **文字识别**：对检测框内的文字内容进行识别，生成最终的文字结果。

本仓库提供的示例根据 PaddleOCR 官方提供的案例，通过模型转换、模型量化和图像后处理后可实际运行可清晰识别字符，可运行 jupyter 脚本文件得到模型推理的结果。

**github 地址**：https://github.com/PaddlePaddle/PaddleOCR

![alt text](../../../resource/imgs/paddleocr.png)

## 2. 性能数据

**RDK X5 & RDK X5 Module**

数据集 ICDAR2019-ArT

| 模型(公版)    | 尺寸(像素)  | 参数量   | BPU吞吐量     |
| ------------ | ------- | ----- | ---------- |
| PP-OCRv3_det | 640x640 | 3.8 M | 158.12 FPS |
| PP-OCRv3_rec | 48x320  | 9.6 M | 245.68 FPS |


**RDK X3 & RDK X3 Module**

数据集 ICDAR2019-ArT

| 模型(公版)    | 尺寸(像素)  | 参数量   | BPU吞吐量     |
| ------------ | ------- | ----- | ---------- |
| PP-OCRv3_det | 640x640 | 3.8 M | 41.96 FPS |
| PP-OCRv3_rec | 48x320  | 9.6 M | 78.92 FPS |


## 3. 模型下载地址

**.bin 文件下载**：

可以使用脚本 [download.sh](./model/download.sh) 一键下载所有此模型结构的 .bin 模型文件：

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x3/en_PP-OCRv3_det_640x640_nv12.bin
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x3/en_PP-OCRv3_rec_48x320_rgb.bin
```

## 4. 部署测试

在下载完毕 .bin 文件后，可以执行 Python/Jupyter 脚本文件，在板端实际运行体验实际测试效果。需要更改测试图片，可额外下载数据集后，放入到data文件夹下并更改 Python/Jupyter 文件中图片的路径

![paddleocr](./data/paddleocr.png)
