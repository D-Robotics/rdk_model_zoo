[English](./README.md) | 简体中文

# EfficientNet-Lite Python 运行示例

Python runtime 包含一个命令行入口和一个可复用 wrapper：

- `main.py`：参数解析、图片和标签读取、配置构造、调用 `predict()`、打印结果。
- `efficientnet.py`：基于 `hbm_runtime` 的 `EfficientNetConfig` 和 `EfficientNet` wrapper。
- `run.sh`：默认运行命令，推理前会下载 sample 内 Lite0 模型。

## 目录结构

```text
runtime/python/
|-- README.md
|-- README_cn.md
|-- efficientnet.py
|-- main.py
`-- run.sh
```

## 环境说明

请在包含 `hbm_runtime`、`numpy` 和 OpenCV 的 RDK S100 Python 环境中运行。本示例复用 `utils/py_utils` 中的公共工具。

## 运行

```bash
bash run.sh
```

## 直接运行入口

```bash
python3 main.py \
  --model-path ../../model/s100/efficientnet_lite0_224x224_nv12.hbm \
  --test-img ../../test_data/Scottish_deerhound.JPEG \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-path` | `../../model/s100/efficientnet_lite0_224x224_nv12.hbm` | 编译后的 HBM 模型路径。 |
| `--test-img` | `../../test_data/Scottish_deerhound.JPEG` | 输入图片路径。 |
| `--label-file` | `../../test_data/imagenet_classes.names` | ImageNet 标签文件。 |
| `--top-k` | `5` | 打印的分类结果数量。 |
| `--resize-type` | `1` | Resize 模式：`0` 拉伸，`1` 保持比例补边。 |
| `--priority` | `0` | 运行时调度优先级。 |
| `--bpu-cores` | `0` | BPU 核心索引。 |

## Wrapper 接口

`EfficientNet` 提供：

- `set_scheduling_params(...)`
- `pre_process(...)`
- `forward(...)`
- `post_process(...)`
- `predict(...)`
- `__call__(...)`

预处理阶段将 BGR 图片转换为 NV12，并按固定输入提供 Y plane 和 UV plane 两个 runtime 输入。
