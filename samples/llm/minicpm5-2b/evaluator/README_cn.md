[English](README.md) | [简体中文](README_cn.md)

# 完整 HBM 评估

在 x86 主机使用[转换环境](../conversion/README_cn.md)。SAMPLE 指向示例目录，WORK 指向工作目录。准备过程需要配套 SDK 和固定 checkpoint，不需要 CUDA；先校验 checkpoint、TEST parquet 与 FP16 embedding 摘要，再用原 SDK 导出 token 和 mask。

```bash
python "$SAMPLE/evaluator/prepare.py" --model-path "$WORK/MiniCPM5-2B" \
  --test-data "$WORK/datasets/wikitext2-test/test-00000-of-00001.parquet" \
  --output-dir "$WORK/ppl-bundle"
scp -r "$WORK/ppl-bundle" user@BOARD:/data/
```

在板端的 evaluator 目录运行：

```bash
python3 -m pip install --target .deps grpcio==1.74.0 protobuf==4.25.8 numpy
EVAL_BUNDLE=/data/ppl-bundle bash run.sh
python3 validate_result.py board-local-ppl.json
# 可选接线检查，不属于完整评估。
EVAL_BUNDLE=/data/ppl-bundle OUTPUT=quick-check.json bash run.sh --samples=1
```

bundle 中的 SDK 协议和 RPC 文件从本机安装的 hbm-infer 3.15.3 提取，不提交到仓库，继续遵循 SDK 条款。脚本启动服务、读取分配端口，并在退出时清理自己的临时目录。MODEL_DIR 默认为 `../model/s600`，OUTPUT 默认为 `board-local-ppl.json`。不要同时加载另一份模型，本次板卡配置不能容纳两份该模型。

## 统计口径与证据

[完整结果](results/s600-wikitext2-full.json)：140 个独立的 2048-token 片段，每段执行八个 256-token chunk，片段间清空 KV。每段 2047 个下一 token 标签全部计入，包含跨 chunk 边界，共 **286580 个预测目标**。总负对数似然除以目标数再取指数得到 PPL，不添加 chat 前缀。4358 行数据用两个换行符拼接后得到 288009 token；所有路径一致排除最后不足 2048 的片段。

最终 PPL **14.242767676160279**，浮点 14.0184，假量化 14.2687，相对上升 **1.60052%**，耗时 **913.192 秒**。JSON 每段更新，未完成 140 段及完成标记前不能视作完整结果。校验脚本检查数学一致性、完整性和 3% 相对 PPL 目标，但不能独立证明实际执行的是哪个模型。

板端使用 NumPy float32 log-softmax 和 float64 损失累加，避免 logits/KV 跨网络传输。前五段 PPL：SDK/PyTorch RPC 15.047808，板端 NumPy 15.048632，相差 0.0055%。交付脚本复现首段 NLL 4907.887529，该接线检查不替代完整测评。

HBM SHA256：`7c54a0934b95c26ec378f93716618f17eb58d3efd5d5b3de7b016040513ed0ee`。

- 精度：hbm-infer 3.15.3、DNN 3.15.3_(4.11.2 HBRT)。
- 生成：OELLM 2.0.4、UCP 3.15.2、DNN 3.15.2_(4.10.6 HBRT)、RDK OS V5.1.0。
- RPC 核心编号为 0,1,2,3，OELLM backends 使用 1,2,3,4。

## 生成验证

[六条提示词及 token 对照](../test_data/generation-reference.json)全部匹配官方 HF 贪心输出，并以 EOS 结束。另有 53/45-token 两轮对话、填充后 2048/3840-token 输入的代码检索和 50 次重复请求。重复测试平均 decode 53.2503 token/s、首 token 延迟 147.8602 ms，不含冷启动加载时间。这是短时单请求场景，不代表老化或并发覆盖；prefill 计数包含 chunk 填充。

## S100 / S100P 全量验证（2026-09-09）

使用独立的 [legacy 评估入口](legacy/README_cn.md)，对应 SDK 1.0.0、UCP/DNN 3.7.3、HBRT 4.2.11。两块板各自的 HBM 均完成相同的 140 × 2048-token TEST 输入与 286580 个预测目标。**两板 PPL 均为 17.91995474675122，相对浮点上升 27.83167%，未达到 ≤3% 精度目标。** 完整性与数学一致性检查通过；共用验证器会按精度阈值拒绝该结果。

| 板卡 | 全量 PPL | 评估秒数 | 参考文本完全一致 | 连续请求 |
|---|---:|---:|---:|---:|
| [S100 PPL](results/s100-wikitext2-full.json) / [生成](results/s100-generation-full.json) | 17.91995 | 1129.83 | 2/6 | 50/50 |
| [S100P PPL](results/s100p-wikitext2-full.json) / [生成](results/s100p-generation-full.json) | 17.91995 | 882.44 | 2/6 | 50/50 |

两板均通过中英双轮，以及约 2000/3750 原始 token 输入的信息检索。每板共 60 次请求均正常结束，SDK 返回码和销毁返回码正常。四条参考文本不一致：翻译错误、代码回答不完整，以及 JSON/列表格式不同，不能算作六条生成对照全部通过。旧版 SDK 只返回文本，不返回生成 token ID。

50 次请求平均端到端耗时为 S100 671.37 ms、S100P 543.22 ms，不含模型加载；这是墙钟时间，不是 TTFT。此前短请求日志 decode 约 12.1/13.0 token/s，不把全零回调字段当成性能测量。未覆盖工具调用、思考、多模态、并发或长时间压力测试。

[首片段诊断](results/legacy-first-segment-diagnostic.json)：HF float32 PPL 10.75668，legacy 适配器 float32 10.75612，实际 HBM 14.01536。全部八个掩码与 SDK 校准辅助函数逐项一致；输入哈希与 S600 相同，也核验了实际执行的两份 HBM 哈希。这将后续排查重点指向量化执行路径，但尚未定位具体量化算子。交付版评估脚本在两板分别复现首段 NLL 5404.3955137729645。上文 S600 结果沿用既有记录，本次没有重新评测。
