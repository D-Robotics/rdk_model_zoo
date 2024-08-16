English| [简体中文](./README_cn.md)

Yolo World
=======

# 1. Model introduction

YOLO (You Only Look Once) is a real-time object detection system whose core concept is to convert the object detection task into a single regression problem. YOLO World is the latest improved version of this model, offering higher accuracy and speed. It introduces an innovative approach to enhance YOLO's open vocabulary detection capabilities through visual language modeling and pre-training on large-scale datasets. Specifically, this method involves a new re-parameterizable Visual-Language Path Aggregation Network (RepVL-PAN) and region-text contrastive loss to facilitate interaction between visual and linguistic information. This approach efficiently detects a wide range of objects in zero-shot scenarios. On the challenging LVIS dataset, YOLO-World achieves an AP of 35.4 at 52.0 FPS on a V100, surpassing many state-of-the-art methods in both accuracy and speed. Additionally, the fine-tuned YOLO-World performs exceptionally well on various downstream tasks, including object detection and open-vocabulary instance segmentation.

# 2. Model download link

- yolo_world.bin: TODO

yolo_world.bin should be in the same directory as the current README.md.

# 3. Input and Output Data

## 3.1 Image Encoder

- Input Data

  | Input Data | Data Type | Shape                            | Layout |
  | -------- | -------- | ------------------------------- | ------------ |
  | image    | FLOAT32  | 1 x 3 x 640 x 640 | NCHW           |
  | text    | FLOAT32  | 1 x 3 x 512 | NCHW           |

- Output Data

  | Input Data | Data Type | Shape                            | Layout |
  | -------- | -------- | ------------------------------- | ------------ |
  | classes_score    | FLOAT32  | 1 x 8400 x 32 | NCHW           |
  | bboxes    | FLOAT32  | 1 x 8400 x 4 | NCHW           |