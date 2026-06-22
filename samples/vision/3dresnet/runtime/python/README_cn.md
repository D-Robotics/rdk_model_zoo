[English](./README.md) | 简体中文

# Python 运行示例

本目录包含 3D ResNet-18 视频动作分类的 Python 运行示例。

## 文件说明

```text
.
|-- README.md
|-- README_cn.md
|-- main.py
|-- resnet3d.py
`-- run.sh
```

## 快速运行

```bash
bash run.sh
```

## 直接运行

```bash
python3 main.py \
  --model-path ../../model/s100/r3d_18.hbm \
  --test-clip ../../test_data/video0.npy \
  --label-file ../../test_data/kinetics_classnames.json \
  --top-k 5
```

## 参数说明

| 参数 | 说明 | 默认值 |
| ---- | ---- | ------ |
| `--model-path` | HBM 模型路径 | `../../model/s100/r3d_18.hbm` |
| `--test-clip` | `.npy` 格式预处理视频片段 | `../../test_data/video0.npy` |
| `--label-file` | Kinetics-400 类别映射 JSON 文件 | `../../test_data/kinetics_classnames.json` |
| `--top-k` | 输出分类数量 | `5` |
| `--priority` | hbm_runtime 调度优先级 | `0` |
| `--bpu-cores` | hbm_runtime 使用的 BPU 核心编号 | `0` |

## 输入输出

模型输入为 shape `(1, 3, 16, 112, 112)` 的 float32 张量。

模型输出为 Kinetics-400 logits。`resnet3d.py` 将 logits 转换为 Top-K 概率，`main.py` 打印类别名称和分数。
