[English](./README.md) | 简体中文

# Python 推理

本目录包含 MobileNetV1 Python 推理入口和模型封装。`main.py` 负责命令行参数、
图片与标签读取、运行配置、调用预测和结果打印；`mobilenetv1.py` 实现
`Config + Wrapper + predict()` 推理流程。

## 文件

```text
.
|-- main.py
|-- mobilenetv1.py
|-- run.sh
|-- README.md
`-- README_cn.md
```

## 参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-path` | HBM 模型路径 | `../../model/s100/mobilenetv1_224x224_nv12.hbm` |
| `--test-img` | 输入图片路径 | `../../test_data/zebra_cls.jpg` |
| `--label-file` | ImageNet 标签文件 | `../../test_data/imagenet_classes.names` |
| `--top-k` | 打印的分类结果数量 | `5` |
| `--priority` | 运行优先级，0 最低 | `0` |
| `--bpu-cores` | BPU 核心编号列表 | `0` |

## 运行

一键运行：

```bash
bash run.sh
```

直接运行：

```bash
python3 main.py \
  --model-path ../../model/s100/mobilenetv1_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

## 预期结果

使用 `zebra_cls.jpg` 时，Top-5 输出应包含 `zebra`，分数为合理非零值，
不应出现 NaN 或全 0 分布。
