[English](./README.md) | 简体中文 | [日本語](./README_jp.md)

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

本仓库提供的示例基于 PaddleOCR 官方提供的算法结构，通过模型转换与量化后，可实现高精度的端侧文字检测与识别。用户可进入 `runtime/python` 运行 Python 脚本得到模型推理的实际结果。

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

可以使用脚本 [download_model.sh](./model/download_model.sh) 一键下载所有此模型结构的 .bin 模型文件：

```shell
cd model
bash download_model.sh
```

## 4. 部署测试

在下载完毕 .bin 文件后，进入 `runtime/python` 目录执行 Python 脚本，以体验板端实际的测试效果：

```shell
# 进入运行目录
cd runtime/python

# 直接使用预设脚本运行体验
bash run.sh

# 或者根据实际需求修改参数后手动运行
python3 main.py --det_model_path ../../model/en_PP-OCRv3_det_640x640_nv12.bin \
                --rec_model_path ../../model/en_PP-OCRv3_rec_48x320_rgb.bin \
                --image_path ../../test_data/paddleocr_test.jpg \
                --output_folder ../../test_data/output/predict.jpg
```

如果需要更改测试图片，可以将自定义图片存放到 `test_data` 文件夹下，并更改执行命令中的 `image_path` 路径参数。

![paddleocr](./test_data/paddleocr.png)
