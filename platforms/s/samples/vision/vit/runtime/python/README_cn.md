[English](./README.md) | 简体中文

# ViT Python 运行示例

Python runtime 包含一个命令行入口和一个可复用 wrapper：

- `main.py`：参数解析、图片和标签读取、配置构造、调用 `predict()`、打印结果。
- `vit.py`：基于 `hbm_runtime` 的 `ViTConfig` 和 `ViT` wrapper。
- `run.sh`：默认运行命令，推理前会下载 sample 内模型。

## 目录结构

```text
runtime/python/
|-- README.md
|-- README_cn.md
|-- main.py
|-- run.sh
`-- vit.py
```

## 环境说明

请在包含 `hbm_runtime`、`numpy` 和 OpenCV 的 RDK S100 Python 环境中运行。本示例复用 `utils/py_utils` 中的公共工具。

## 运行

```bash
bash run.sh int8
```

使用 int16 模型：

```bash
bash run.sh int16
```

## 直接运行入口

```bash
python3 main.py \
  --model-path ../../model/s100/vit_cifar10_batch1_int8.hbm \
  --test-img ../../test_data/airplane_0000.png \
  --label-file ../../test_data/cifar10_classes.names \
  --top-k 5
```

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-variant` | `int8` | 模型变体，可选 `int8` 或 `int16`。 |
| `--model-path` | `../../model/s100/vit_cifar10_batch1_int8.hbm` | 编译后的 HBM 模型路径。 |
| `--test-img` | `../../test_data/airplane_0000.png` | 输入图片路径。 |
| `--label-file` | `../../test_data/cifar10_classes.names` | CIFAR-10 标签文件。 |
| `--top-k` | `5` | 打印的分类结果数量。 |
| `--priority` | `0` | 运行时调度优先级。 |
| `--bpu-cores` | `0` | BPU 核心索引。 |

## Wrapper 接口

`ViT` 提供：

- `set_scheduling_params(...)`
- `pre_process(...)`
- `forward(...)`
- `post_process(...)`
- `predict(...)`
- `__call__(...)`

预处理阶段将 BGR 图片转换为 NV12，并按固定输入提供 Y plane 和 UV plane 两个 runtime 输入。
