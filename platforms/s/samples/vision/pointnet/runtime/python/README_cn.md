[English](./README.md) | 简体中文

# Python 运行示例

本目录包含 PointNet 椅子部件分割的 Python 运行示例。

## 文件说明

```text
.
|-- README.md
|-- README_cn.md
|-- main.py
|-- pointnet.py
`-- run.sh
```

## 依赖

Runtime 使用 `numpy`、`matplotlib` 和 `hbm_runtime`。

## 快速运行

```bash
bash run.sh
```

## 直接运行

```bash
python3 main.py \
  --model-path ../../model/s100/pointnet.hbm \
  --test-pts ../../test_data/chair.pts \
  --img-save-path result.png
```

## 参数说明

| 参数 | 说明 | 默认值 |
| ---- | ---- | ------ |
| `--model-path` | HBM 模型路径 | `../../model/s100/pointnet.hbm` |
| `--test-pts` | `.pts` 格式输入点云 | `../../test_data/chair.pts` |
| `--img-save-path` | 分割可视化结果保存路径 | `result.png` |
| `--priority` | hbm_runtime 调度优先级 | `0` |
| `--bpu-cores` | hbm_runtime 使用的 BPU 核心编号 | `0` |

## 输入输出

`pointnet.py` 读取 `chair.pts`，对坐标进行中心化和尺度归一化，并转置为 `(1, 3, N)` 输入。

模型输出为 shape `(1, N, 4)` 的逐点 logits。Runtime 会打印 `back`、`seat`、`leg`、`arm` 的点数，并保存原始点云图和分割结果图。
