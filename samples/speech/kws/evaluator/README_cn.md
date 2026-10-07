# KWS evaluation

[English](README.md) | 简体中文

<a id="dataset"></a>
## 数据集

随附片段是一个“hey snips”正例。准备独立的正负片段和稳定 ID，保持单声道 16 kHz 前端及分窗规则一致；长录音需显式切窗并记录策略。

评估器读取已保存的概率。输入 JSON schema 为 `rdk-model-zoo/kws-predictions/v1`；`provenance` 包含非空 `dataset`、`model`、`split` 描述，`records` 含唯一 `id`、整数 `label` 0/1 和有限 [0,1] `score`。采集预测时，在 provenance 中补充模型/输入摘要和前端版本。

<a id="environment"></a>
## 环境

计分仅需 Python 3.10+ 标准库，不需 SDK、音频库或 NumPy。推理依赖和 metadata 检查见[运行说明](../runtime/python/README_cn.md)。

<a id="command"></a>
## 命令

下面示例创建两条记录以展示输入 schema。使用新的临时目录：

```bash
KWS_EVAL_DIR=$(mktemp -d)
cat > "$KWS_EVAL_DIR/predictions.json" <<'JSON'
{"schema":"rdk-model-zoo/kws-predictions/v1","provenance":{"dataset":"synthetic documentation example","model":"no model executed","split":"fixture"},"records":[{"id":"positive","label":1,"score":0.9},{"id":"negative","label":0,"score":0.1}]}
JSON
python3 samples/speech/kws/evaluator/evaluate.py \
  --predictions "$KWS_EVAL_DIR/predictions.json" --threshold 0.5 \
  --output "$KWS_EVAL_DIR/metrics.json"
```

测量时换成实际采集的预测和来源信息。`--output` 必填且不能已存在，父目录可自动创建；`--threshold` 默认 0.5。非法记录、重复 ID、空集合、非有限/越界分数或缺 provenance 都退出 2。

<a id="metrics"></a>
## 指标

预测正例条件为分数大于等于阈值。报告 TP/TN/FP/FN、accuracy、precision、recall、F1、误接受率 `FP/(FP+TN)` 和漏检率 `FN/(FN+TP)`。分母为零时返回 JSON null。指标按片段/窗口统计；每小时事件率需事件时间戳和聚合计算，输入 schema 不含这些字段。评估器不计算 ROC 曲线。

<a id="outputs"></a>
## 输出

`metrics.json` 使用 `rdk-model-zoo/kws-metrics/v1`，保留 provenance 和精确预测文件 SHA-256。schema 字段 `inference_executed=false`、`provenance_independently_verified=false`。示例记录得到两条正确判定。请和报告一起保存输入预测文件。

<a id="reference-results"></a>
## 源参考结果

S100 参考测量：

| 范围 | 源数值 |
| --- | --- |
| S100 `hrt_model_exec perf`，100 帧 | 平均延迟 1.176 ms；830.875 FPS |
| 随附“hey snips”音频 | 约 0.985 置信度 |

源性能命令为 `hrt_model_exec perf --model_file /root/kws/kws.hbm --frame_count 100`，测量 HBM 执行的 100 帧；音频读取和特征提取不包含在该命令中。

<a id="boundaries"></a>
## 适用范围

这些指标描述片段/窗口判定。每小时事件率需要事件时间戳和聚合；板端延迟由 S100 运行流程记录。每组预测应记录代码提交、目标、SDK、模型/输入摘要和前端版本。
