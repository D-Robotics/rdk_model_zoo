[English](./README.md) | 简体中文

# PointNet 模型说明

PointNet 是用于点云处理的神经网络。它直接接收无序 3D 点，使用共享 MLP 提取点特征，并通过对称的 max 操作聚合全局特征。本 sample 使用 PointNet 椅子部件分割模型，将每个输入点分到 back、seat、leg、arm 四类之一。

测试输入为 `chair.pts` 点云。运行时会保存原始点云视图和预测分割视图。

## 算法介绍

PointNet 是直接处理无序点集的深度学习模型，常用于 3D 分类和点云分割任务。

- **论文**: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- **参考实现**: [charlesq34/pointnet](https://github.com/charlesq34/pointnet)

## 算法功能

- 椅子点云部件分割
- 输出每个点所属的部件标签

## 算法特点

- **无序点集输入**：直接处理 `N x 3` 点云坐标。
- **共享 MLP**：对每个点提取局部特征。
- **对称聚合**：通过 max 操作获得顺序无关的全局特征。

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
|       |-- pointnet.py
|       `-- run.sh
`-- test_data
    |-- chair.pts
    `-- readme_img
```

## 快速体验

```bash
cd samples/vision/pointnet/runtime/python
bash run.sh
```

脚本会检查 `../../model/s100/pointnet.hbm`，使用 `../../test_data/chair.pts` 执行分割，并保存 `result_orig.png` 和 `result.png`。

## 模型转换

- HBM 模型文件见 [model](./model/README_cn.md) 目录。
- 转换说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 模型推理

本 sample 当前维护 Python 推理路径，详细说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

| 模型 | 任务 | 输入 | 输出 |
| ---- | ---- | ---- | ---- |
| PointNet | 椅子部件分割 | 归一化点云 `(1, 3, N)` | 逐点 logits `(1, N, 4)` |

## 模型评估

评测说明、性能记录和结果检查方法请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 推理结果

使用 `chair.pts` 时，输出应包含 `back`、`seat`、`leg`、`arm` 四类部件点数，并保存原始点云图和分割结果图。

## License

本 sample 遵循 [Apache 2.0 License](../../../LICENSE)。
