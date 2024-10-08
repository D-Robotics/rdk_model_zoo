[English](./README.md) | 简体中文

Yolo World
=======

# 1. 模型介绍

YOLO (You Only Look Once) 是一种实时目标检测系统，其核心理念是将目标检测任务转换为单次回归问题。YOLO World 是该模型的最新改进版本，具有更高的准确性和速度。这是一种通过视觉语言建模和大规模数据集的预训练来增强 YOLO 开放词汇检测能力的创新方法。具体来说，这是一种新的可重新参数化的视觉语言路径聚合网络 (RepVL-PAN) 和区域文本对比损失，以促进视觉信息和语言信息之间的交互。该方法在零样本情况下高效地检测到广泛的对象。在具有挑战性的 LVIS 数据集上，YOLO-World 在 V100 上以 52.0 FPS 实现了 35.4 的 AP，超越了许多最新的先进方法，无论在准确性还是速度方面。此外，微调后的 YOLO-World 在多个下游任务中表现出色，包括目标检测和开放词汇实例分割。

# 2. 模型下载地址

地瓜异构.bin模型文件已经上传至云服务器中，可通过 wget 命令在服务器网站中下载：

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolo_world.bin
```

将yolo_world.bin放入与当前README.md的同级目录即可。

# 3. 输入输出数据


- 输入数据

  | 输入数据 | 数据类型 | 大小                            | 数据排布格式 |
  | -------- | -------- | ------------------------------- | ------------ |
  | image    | FLOAT32  | 1 x 3 x 640 x 640 | NCHW           |
  | text    | FLOAT32  | 1 x 3 x 512 | NCHW           |
  

- 输出数据

  | 输出数据 | 数据类型 | 大小                            | 数据排布格式 |
  | -------- | -------- | ------------------------------- | ------------ |
  | classes_score    | FLOAT32  | 1 x 8400 x 32 | NCHW           |
  | bboxes    | FLOAT32  | 1 x 8400 x 4 | NCHW           |
