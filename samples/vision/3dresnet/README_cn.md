[English](./README.md) | 简体中文

# 3D ResNet-18 模型说明

3D ResNet-18 (R3D-18) 是视频动作分类模型。它将 2D ResNet 扩展为 3D 卷积结构，从短视频片段中同时提取空间和时间特征。本 sample 运行一个预处理后的 16 帧视频片段，并输出 Kinetics-400 Top-K 动作分类结果。

当前测试输入为 `video0.npy`。对于该片段，合理结果应将动作分类为 `archery`。

## 算法介绍

R3D-18 将 ResNet18 的二维卷积扩展为三维卷积，用于同时建模视频帧的空间特征和时间特征。

- **论文**: [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248)
- **参考实现**: [torchvision r3d_18](https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html)

### 算法功能

- Kinetics-400 视频动作分类
- Top-K 动作类别输出

### 算法特点

- **3D 卷积**：在空间和时间维度上同时提取特征。
- **残差结构**：沿用 ResNet 残差连接，降低深层网络训练难度。
- **预处理输入**：runtime 使用 `.npy` 视频片段作为输入。

## 目录结构

```text
.
|-- README.md
|-- README_cn.md
|-- conversion
|   |-- README.md
|   `-- README_cn.md
|-- evaluator
|   |-- README.md
|   `-- README_cn.md
|-- model
|   |-- README.md
|   |-- README_cn.md
|   `-- download_model.sh
|-- runtime
|   `-- python
|       |-- README.md
|       |-- README_cn.md
|       |-- main.py
|       |-- resnet3d.py
|       `-- run.sh
`-- test_data
    |-- kinetics_classnames.json
    |-- readme_img
    `-- video0.npy
```

## 快速体验

```bash
cd runtime/python
bash run.sh
```

脚本会在需要时下载 `../../model/s100/r3d_18.hbm`，并使用 `../../test_data/video0.npy` 执行推理。

## 模型转换

- 预编译 HBM 模型通过 [model](./model/README_cn.md) 目录提供。
- 转换说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 模型推理

本 sample 当前维护 Python 推理路径，详细说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

| 模型 | 任务 | 输入 | 输出 |
| ---- | ---- | ---- | ---- |
| R3D-18 | 视频动作分类 | `(1, 3, 16, 112, 112)` float32 视频片段 | Kinetics-400 logits |

## 模型评估

评测说明、性能记录和结果检查方法请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 推理结果

使用 `video0.npy` 时，合理结果应将 Top-1 预测为 `archery`，Top-5 中其余类别分数明显更低。

## License

遵循 Model Zoo 顶层 License。
