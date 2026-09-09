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

## S100 / S100P 基础验证

上面的完整 PPL 评估入口仅用于 S600。S100/S100P 当前只有中英文单轮生成记录（2026-09-09，W8/chunk256/cache4096，SDK 1.0.0，DNN 3.7.3/HBRT 4.2.11）。

| 板卡 | Prefill token/s | Decode token/s |
|---|---:|---:|
| S100 | 431.70–432.43 | 12.07–12.11 |
| S100P | 554.11 | 12.97–13.04 |

数据取自短请求 Runtime Performance 日志；prefill 按 256-token 分块补齐。回调性能字段为零，不作为 TTFT。没有全量 PPL、多轮、工具调用、多模态、长上下文或老化结论。

```bash
cd ../runtime/legacy
BOARD=s100 bash run.sh --prompt 'What is the capital of France?'
BOARD=s100 bash run.sh --prompt '请用一句话介绍你自己。'
# 在 S100P 上改用 BOARD=s100p。
```

英文预期 `The capital of France is Paris.`；中文应介绍 MiniCPM 与 ModelBest/OpenBMB。两者必须以 `RESULT status=0 ended=1 failed=0 destroy=0` 结束，且没有模板回退或 `<|im_end|>` 外泄。这是基础功能验证，不是语言模型质量基准。
