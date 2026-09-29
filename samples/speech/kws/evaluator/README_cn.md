# KWS evaluation

[English](README.md) | 简体中文

<a id="dataset"></a>
## 数据集

随附片段只有一个“hey snips”正例，不是完整评估集。准备独立正负片段、稳定 ID，并保持单声道 16 kHz 前端和分窗规则一致；长录音须显式切窗并记录策略。不能在同一划分调阈值并报告结果而不披露。

离线评估器读取已保存概率，不读取音频或 HBM。输入 JSON schema 为 `rdk-model-zoo/kws-predictions/v1`；`provenance` 包含非空 `dataset`、`model`、`split` 描述，`records` 含唯一 `id`、整数 `label` 0/1 和有限 [0,1] `score`。采集真实预测时，在 provenance 中补充模型/输入摘要和前端版本；评分器保留这些信息，但不独立认证其真实性。

<a id="environment"></a>
## 环境

仅需 Python 3.10+ 标准库，不需 SDK、音频库或 NumPy。推理依赖和 metadata 检查见[运行说明](../runtime/python/README_cn.md)。可以在主机对其他环境采集的证据计分，不会暗中运行模型。

<a id="command"></a>
## 命令

下面主机示例明确写入合成分数，只演示 schema 和计算，不测量发布模型。使用新的临时目录：

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

预测正例条件为分数大于等于阈值。报告 TP/TN/FP/FN、accuracy、precision、recall、F1、误接受率 `FP/(FP+TN)` 和漏检率 `FN/(FN+TP)`。分母为零返回 JSON null，不虚构满分。这是每片段/窗口的比例，**不是每小时误唤醒次数**；不计算事件时间或 ROC 曲线。

<a id="outputs"></a>
## 输出

`metrics.json` 使用 `rdk-model-zoo/kws-metrics/v1`，保留 provenance 和精确预测文件 SHA-256。`inference_executed=false`、`provenance_independently_verified=false` 明确验证边界。文档夹具的两条判定都正确，但不是数据集准确率；请和报告一起保存输入预测文件。

<a id="reference-results"></a>
## 源参考结果

[归档 S evaluator](../../../../platforms/s/samples/speech/kws/evaluator/README_cn.md)记录：

| 范围 | 源数值 | 当前状态 |
| --- | --- | --- |
| S100 `hrt_model_exec perf`，100 帧 | 平均延迟 1.176 ms；830.875 FPS | 历史记录，未重跑 |
| 随附“hey snips”音频 | 约 0.985 置信度 | 历史功能示例，未重跑 |

源性能命令为 `hrt_model_exec perf --model_file /root/kws/kws.hbm --frame_count 100`，不是完整音频/文件/前处理耗时。不要由延迟倒算新 FPS，也不要声称两项统计范围相同。本轮主机证据覆盖特征一致性、契约和指标计算，没有新增真实 HBM 分数。

<a id="boundaries"></a>
## 边界

本迁移未建立数据集 precision/recall、阈值校准、板端延迟、流式事件准确率或 S100P/S600 行为。历史值不代表重构代码验收；后续板端对照须绑定精确代码、SDK、模型/输入摘要、前端版本、命令和完整输出。
