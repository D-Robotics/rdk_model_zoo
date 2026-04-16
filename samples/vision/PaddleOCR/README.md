English | [简体中文](./README_cn.md) | [日本語](./README_jp.md)

# PaddleOCR text recognition

- [PaddleOCR text recognition](#paddleocr-text-recognition)
  - [1. Introduction to PaddleOCR](#1-introduction-to-paddleocr)
  - [2. Performance data](#2-performance-data)
  - [3. Model download link](#3-model-download-link)
  - [4. Deploy tests](#4-deploy-tests)


## 1. Introduction to PaddleOCR

PaddleOCR is Baidu PaddlePaddle's Optical Character Recognition (OCR) tool based on Deep learning. It uses the PaddlePaddle framework to perform text recognition tasks in images. The repository converts text in images into editable text through multiple stages such as image preprocessing, text detection, and text recognition. PaddleOCR supports recognition of multiple languages and fonts, suitable for text extraction tasks in various complex scenarios. PaddleOCR also supports custom training, allowing users to prepare training data according to specific needs to further optimize model performance.

In practical applications, the workflow of PaddleOCR includes the following steps:

- **Image preprocessing**: denoising and resizing the input image to make it suitable for subsequent detection and recognition.
- **Text detection**: Detect text areas in images through deep learning models to generate detection boxes.
- **Text recognition**: Recognize the text content in the detection box and generate the final text result.

The examples provided in this repository are based on the algorithm structure from the official PaddleOCR. After model conversion and quantization, high-precision on-device text detection and recognition can be achieved. You can navigate to `runtime/python` and execute the Python script to obtain the inference results.

GitHub: https://github.com/PaddlePaddle/PaddleOCR

![alt text](../../../resource/imgs/paddleocr.png)


## 2. Performance data

**RDK X5 & RDK X5 Module**

Dataset ICDAR2019-ArT

| Model(public) | size(pixels) | Parameter | BPU throughput |
| ------------ | ------- | ----- | ---------- |
| PP-OCRv3_det | 640x640 | 3.8 M | 158.12 FPS |
| PP-OCRv3_rec | 48x320  | 9.6 M | 245.68 FPS |


**RDK X3 & RDK X3 Module**

Dataset ICDAR2019-ArT

| Model(public) | size(pixels) | Parameter | BPU throughput |
| ------------ | ------- | ----- | ---------- |
| PP-OCRv3_det | 640x640 | 3.8 M | 41.96 FPS |
| PP-OCRv3_rec | 48x320  | 9.6 M | 78.92 FPS |


## 3. Model download link

**.Bin file download**:

You can download all .bin model files for this model structure with one click using the script [download_model.sh](./model/download_model.sh):

```shell
cd model
bash download_model.sh
```

## 4. Deploy tests

After downloading the .bin file, navigate to the `runtime/python` directory and execute the Python script to experience the actual testing effect on the board:

```shell
# Navigate to the runtime python directory
cd runtime/python

# Use the default script to run directly
bash run.sh

# Or you can run the primary script manually configuring the parameters
python3 main.py --det_model_path ../../model/en_PP-OCRv3_det_640x640_nv12.bin \
                --rec_model_path ../../model/en_PP-OCRv3_rec_48x320_rgb.bin \
                --image_path ../../test_data/paddleocr_test.jpg \
                --output_folder ../../test_data/output/predict.jpg
```

To change the test image, place your custom images in the `test_data` folder and adjust the `image_path` parameter in your execution command.

![paddleocr](./test_data/paddleocr.png)
