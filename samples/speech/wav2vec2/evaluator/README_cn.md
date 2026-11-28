# 模型评测

[English](./README.md) | 简体中文

本目录提供 Wav2Vec2 sample 的 benchmark 说明和验证参考。

## 支持模型

| 模型 | 参数量 (M) | 任务 |
| --- | --- | --- |
| Wav2Vec2 Base | 95 | ASR |

## 测试环境

- 平台：`RDK X5`
- 运行时后端：`hbm_runtime`
- 模型格式：`.bin`
- CPU：8xA55@1.8GHz，全核 Performance 调度
- BPU：1xBayes-e@1GHz，等效 10TOPS INT8 算力

## 指标说明

- **WER**：词错误率，ASR 评估的标准指标。
- **延迟**：单次音频输入的端到端推理延迟。
- **RTF**：实时因子，推理时间与音频时长的比值。

## Benchmark 结果

| 模型 | WER (%) | 延迟 (ms) | RTF |
| --- | --- | --- | --- |
| Wav2Vec2 Base | 待定 | 待定 | 待定 |

## 运行评测

准备好 `model/` 中的部署文件后，在板端的本目录执行：

```bash
python3 eval.py
```

默认评测 `../test_data/test_audio.wav`，参考文本来自 `../test_data/reference.json`。传入 `--audio-dir` 和 `--ground-truth` 可评测自己的数据集。`--json-save-path` 指定逐条转录与平均 WER 的输出文件。
