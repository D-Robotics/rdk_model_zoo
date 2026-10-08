[English](README.md) | 简体中文

# S100 / S100P 全量评估

使用对应板卡的 SDK 1.0.0 HBM 与运行库。评估通过板端
`hbm_runtime.HB_HBMRuntime` 执行，不使用 S600 RPC 包。实测板卡已有
Python 3.10、NumPy 和 `hbm_runtime`；生成验证还需要 g++ 和
`nlohmann/json.hpp`（Ubuntu 包 `nlohmann-json3-dev`）。必须在 Python
导入运行库前设置 SDK 库路径。同一块板上不要同时执行 PPL 与生成测试。

## 目录结构

```text
legacy/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── acceptance.cc  # 源码或数据文件
├── evaluate.py  # Python 脚本
├── prepare.py  # Python 脚本
├── run.sh  # 运行示例
└── run_acceptance.sh  # Shell 脚本
```

## 主机准备统一输入

在安装了 NumPy、datasets、transformers 的环境中，使用转换说明中固定的
原始 MiniCPM5-2B tokenizer 与 WikiText2 TEST parquet：

```bash
python prepare.py --model-path /data/MiniCPM5-2B \
  --test-data /data/wikitext2-test/test-00000-of-00001.parquet \
  --output-dir /data/legacy-ppl-input
scp -r /data/legacy-ppl-input user@BOARD:/data/
```

脚本校验 TEST parquet 以及与 S600 完全一致的 token 文件哈希，不依赖 S600
SDK。也可以直接复用已校验的 S600 `input_ids.npy`。PPL 不添加聊天模板。

## 分别在两块板上执行

```bash
export BOARD=s100  # S100P 改成 s100p
export OELLM_SDK_ROOT=/data/D-Robotics_LLM_S100_1.0.0_SDK
export EVAL_BUNDLE=/data/legacy-ppl-input
bash run.sh
python3 ../validate_result.py legacy-ppl.json
# 可选：单片段接线检查，不满足全量评估要求。
OUTPUT=partial.json bash run.sh --samples 1

# PPL 进程退出后，再执行生成测试。
bash run_acceptance.sh | tee acceptance.log
```

下载器会校验压缩包及解压成员；评估脚本另外记录实际执行 HBM 的哈希。
每个片段完成后原子更新结果。只有 140 段、286580 个预测目标与
`FULL_EVALUATION_COMPLETE` 标记才代表全量完成。共用的验证器另外检查相对
浮点参考 14.0184 的 PPL 上升是否 ≤3%；测试完成本身不说明精度达标。

## 数值口径与生成验证范围

PPL 使用 140 个独立的 2048-token 片段，每段 8 个 256-token chunk，段间
清空 KV。每段计入全部 2047 个下一 token 标签，包括 chunk 边界。按 HBM
尺度和零点还原 logits；K/V 保留各自的整数类型，检查输出与输入尺度一致。
旧版缓存沿第 0 维滚动。log-sum-exp 使用 float32，损失累加使用 float64。

生成验证沿用 legacy demo 的贪心参数和非思考模板，并保留一个 SDK 句柄
测试多轮和重复请求。覆盖 6 条 HF 参考文本、中英双轮、约 2000/3750 原始
token 长输入检索，以及 50 次新会话请求。记录每条文本比较、SDK 返回码、
正常结束/错误事件和实际耗时。旧版 SDK 不提供生成 token ID，因此文本
比较不等同于 S600 的 token ID 逐项比较；不把全零回调字段当成 TTFT 或
decode 速度。并发、长时间稳定性、工具调用、思考与多模态不在本评估范围内，
部署前请另行验证。


## 短请求生成吞吐量

条件为 S100/S100P、W8、chunk 256、cache 4096、SDK 1.0.0、DNN 3.7.3/HBRT 4.2.11。单轮中英文请求数据不含冷启动模型加载；prefill 数量包含补齐到 256 token 的部分。

| Board | Prefill tokens/s | Decode tokens/s |
|---|---:|---:|
| S100 | 431.70–432.43 | 12.07–12.11 |
| S100P | 554.11 | 12.97–13.04 |
