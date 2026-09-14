[English](./README.md) | 简体中文

# ResNet152 Python 运行示例

Python runtime 包含一个轻量入口脚本和一个可复用 wrapper：

- `main.py`：参数解析、图片和标签读取、配置构造、调用 `predict()`、打印结果。
- `resnet152.py`：基于 `hbm_runtime` 的 `Resnet152Config` 和 `Resnet152` wrapper。
- `run.sh`：默认运行命令，推理前会下载 sample 内模型。

## 目录结构

```text
runtime/python/
|-- README.md
|-- README_cn.md
|-- main.py
|-- resnet152.py
`-- run.sh
```

## 环境说明

请在包含 `hbm_runtime`、`numpy` 和 OpenCV 的 RDK Python 环境（S100 或 S600）中运行。本示例复用 `utils/py_utils` 中的公共工具。

## 运行

```bash
bash run.sh
```

`run.sh` 默认下载并使用 S100 模型。RDK S600 用户请先执行
`bash ../../model/download_model.sh s600`，并将 `--model-path` 改为
`../../model/s600/resnet152_224x224_nv12.hbm`。

## 直接运行入口

```bash
python3 main.py \
  --model-path ../../model/s100/resnet152_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../../../../datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-path` | `../../model/s100/resnet152_224x224_nv12.hbm` | 编译后的 HBM 模型路径。 |
| `--test-img` | `../../test_data/zebra_cls.jpg` | 输入图片路径。 |
| `--label-file` | `../../../../../datasets/imagenet/imagenet_classes.names` | ImageNet 标签文件。 |
| `--top-k` | `5` | 打印的分类结果数量。 |
| `--priority` | `0` | 运行时调度优先级。 |
| `--bpu-cores` | `0` | BPU 核心索引。 |

## Wrapper 接口

`Resnet152` 提供：

- `set_scheduling_params(...)`
- `pre_process(...)`
- `forward(...)`
- `post_process(...)`
- `predict(...)`
- `__call__(...)`

预处理阶段将 BGR 图片转换为 NV12，并按固定输入提供 Y plane 和 UV plane 两个 runtime 输入。
