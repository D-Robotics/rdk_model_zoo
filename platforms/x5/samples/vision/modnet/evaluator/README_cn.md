# 模型评估

本目录包含用于评估 MODNet 模型精度和性能的说明。

## 目录结构

```text
.
├── README.md              # 使用说明 (英文)
└── README_cn.md           # 使用说明 (中文)
```

## 评估类型

| 类型 | 说明 | 工具 |
|------|------|------|
| 性能评估 | 测试推理速度和吞吐量 | `hrt_model_exec perf` |
| 精度评估 | 在抠图基准数据集上评估 alpha matte 质量 | 通用评估脚本 |

## 性能评估

使用 `hrt_model_exec` 测量推理延迟和吞吐量：

```bash
hrt_model_exec perf \
    --model_file ../../model/modnet_512x512_rgb.bin \
    --thread_num 1
```

## 精度评估

使用 tools 目录下的通用评估脚本在抠图基准数据集上进行精度评估。

有关更多详细信息，请参考 [通用评估指南](../../../../../docs/README.md)。
