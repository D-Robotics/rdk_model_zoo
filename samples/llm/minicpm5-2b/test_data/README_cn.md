[English](README.md) | 简体中文

# 生成参考数据


本目录包含文本生成提示词与参考结果。

## 目录结构

```text
test_data/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── generation-reference.json  # 结构化数据
├── legacy-long-prompts.json  # 结构化数据
├── legacy-prompts.json  # 结构化数据
└── prompts.json  # 结构化数据
```

| 文件 | 用途与适用目标 |
| --- | --- |
| `prompts.json` | 六条示例提示词 |
| `generation-reference.json` | 官方 HF greedy 文本/token IDs 与源 S600 输出的参考记录 |
| `legacy-prompts.json` | SDK 1.0.0 的单轮/双轮生成用例 |
| `legacy-long-prompts.json` | SDK 1.0.0 约 2000/3750 token 的信息检索提示词 |

S600 参考 token 列表移除了 EOS，因为 OELLM 分别返回内容 token 与结束状态。原始模型生成的 Markdown 围栏保留，
不能把 JSON/Python 答案改写成源模型没有给出的裸格式。这组用例作为确定性生成检查，补充完整 WikiText2 PPL 评估。

S100/S100P 的独立结果见 [S100](../evaluator/results/s100-generation-full.json) 和
[S100P](../evaluator/results/s100p-generation-full.json)：只有 2/6 参考文本匹配。旧 SDK 不提供生成 token IDs，
因此不能把这些比较描述为 S600 式 token 一致性。完整判据见[评估说明](../evaluator/README_cn.md)。
