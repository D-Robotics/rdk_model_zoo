# ASR transcript evaluation

[English](README.md) | 简体中文

<a id="dataset"></a>
## 数据集
本评测器只给已保存的参考文本/预测文本配对计分。仓库未提供标注语料，本轮没有测量数据集精度。每条语音使用唯一 ID，保留数据集版本、划分、许可及模型/解码来源。运行时分块报告不是标注数据集；应先将完整转录与独立参考文本对齐，再生成下列输入。

<a id="environment"></a>
## 环境
仅需 Python 3.10+ 标准库，无需板卡、SDK、NumPy 或下载模型。评测器不推理；输入来源说明会保留，但不会被独立认证。

<a id="command"></a>
## 命令
```bash
# cwd: repository root; synthetic example, not model predictions
asr_eval_dir=$(mktemp -d)
cat > "$asr_eval_dir/transcripts.json" <<'JSON'
{
  "schema": "rdk-model-zoo/asr-transcripts/v1",
  "provenance": {"dataset": "synthetic-example", "model": "none", "decode_mode": "ctc"},
  "records": [
    {"id": "one", "reference": "你好", "hypothesis": "你号"},
    {"id": "two", "reference": "世界", "hypothesis": "世界"}
  ]
}
JSON
python3 samples/speech/asr/evaluator/evaluate.py --predictions "$asr_eval_dir/transcripts.json" --output "$asr_eval_dir/metrics.json"
# Expected: count=2, reference_characters=4, substitutions=1, cer=0.25
```
`--predictions` 与 `--output` 均为必需路径，无默认值。已有输出文件会被拒绝，避免覆盖旧报告。输入需符合上述 schema，provenance 的 `dataset`、`model`、`decode_mode` 必须为非空字符串；records 必须为非空列表。每条记录需唯一非空 `id`，以及字符串 `reference`、`hypothesis`；文本本身允许为空。

<a id="metrics"></a>
## 指标
CER 为全部语音的 `(替换 + 删除 + 插入) / 参考字符总数`，不是逐条 CER 的平均值。字符按 Python Unicode 码点计算，保留空白、标点、大小写；不归一化、不删除 token、不转换词边界，组合字符的各码点单独计数。所有参考文本为空时 CER 为 `null`，插入错误仍计数。CER 可以大于 1。编辑距离平局优先对角线，其次删除、最后插入，以保证错误分解可复现。

<a id="outputs"></a>
## 输出
独占创建的 JSON 使用 `rdk-model-zoo/asr-metrics/v1`，包含 `count`、`reference_characters`、`exact_matches`、`cer`、汇总 `errors`、逐条 `utterances`、归一化政策、输入 SHA-256 与原样 provenance。`inference_executed`、`provenance_independently_verified` 均为 false。退出 0 只代表计分完成，不代表识别质量通过；输入错误或输出已存在返回 2。

<a id="reference-results"></a>
## 参考结果
S 源分支记录 S100 命令 `hrt_model_exec perf --model_file asr.hbm --frame_count 100`：100 帧、平均延迟 34.426 ms、29.008 FPS。这是历史模型执行性能，不是迁移后的端到端音频延迟、S600 性能或本轮复测。源资料未完整记录 SDK/工具链/模型摘要。

![历史性能](../test_data/readme_img/perf.jpg)
![历史转录](../test_data/readme_img/print.jpg)
![历史量化比较](../test_data/readme_img/acc.jpg)

精度插图是历史量化比较，不是语料 CER。源文档的“前三秒”已纠正：统一入口处理完整 4.59 秒录音，共三个独立窗口。[迁移证据](../../../../docs/releases/unified-migration/evidence/2026-09-28-b10-asr-core/)记录主机测试与前端数值比较；不能从 SDK 替身输出推断真实转录结果。

<a id="boundaries"></a>
## 边界
CTC 和 legacy 对相同 logits 也可能得到不同文本，必须分开报告。分块、重采样和末块补零同样影响转录。离线计分不能证明运行时正确、量化等价、无真实预测时的模型精度或延迟。板端/模型执行、OE 转换及语料评测均未执行。
