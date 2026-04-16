# 模型评估

本目录包含用于评估 ConvNeXt 模型精度和性能的说明。

## 目录结构

```text
.
├── README.md              # 使用说明 (英文)
└── README_cn.md           # 使用说明 (中文)
```

## 评估类型

| 类型 | 说明 | 工具 |
|------|-------------|------|
| 性能评估 | 测试推理速度和吞吐量 | `hrt_model_exec perf` |
| 精度评估 | 在 ImageNet 上计算 Top-1 和 Top-5 精度 | 通用 `eval_classify.py` |

## 精度评估

使用 tools 目录下的通用评估脚本：

```bash
python3 ../../../../tools/eval_batch_python.py \
    --model-path ../../model/ConvNeXt_nano.bin \
    --image-path /path/to/imagenet/val
```

有关更多详细信息，请参考 [通用评估指南](../../../../../docs/README.md)。
