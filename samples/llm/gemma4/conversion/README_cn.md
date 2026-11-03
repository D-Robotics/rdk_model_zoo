# SDK 2.0 转换

[English](README.md) · [Sample](../README_cn.md)

使用已有 SDK 容器环境。原始模型固定为
`google/gemma-4-E2B-it-qat-q4_0-unquantized`，revision
`6befbaca7398925921802abd1f277b495b78b738`。本入口读取 SDK 已发布的
`gemma4_e2b/lm_w4_kv_w8` AutoClip/GPTQ recipe 和 `configs/gemma4.yml`，
只绑定本地路径、语言分支与 S600 编译规格，不复制或修改 SDK 量化实现。

```bash
python3 samples/llm/gemma4/conversion/main.py prepare \
  --sdk-root /opt/D-Robotics_LLM_2.0.0_S600_S100_SDK \
  --model-path /data/gemma-4-E2B-it-qat-q4_0-unquantized \
  --wikitext-dir /data/wikitext-2-raw-v1 --mmstar /data/MMStar.tsv \
  --output /data/gemma4-conversion
```

生成独立 `configs/` 与 `plan.json`。随后按顺序执行 SDK 原生命令；每条只运行
指定阶段，已有实验目录不会被准备入口覆盖：

```bash
python3 samples/llm/gemma4/conversion/main.py run --work-dir /data/gemma4-conversion --stage autoclip
python3 samples/llm/gemma4/conversion/main.py run --work-dir /data/gemma4-conversion --stage gptq
python3 samples/llm/gemma4/conversion/main.py run --work-dir /data/gemma4-conversion --stage float
python3 samples/llm/gemma4/conversion/main.py run --work-dir /data/gemma4-conversion --stage calib
python3 samples/llm/gemma4/conversion/main.py run --work-dir /data/gemma4-conversion --stage eval
python3 samples/llm/gemma4/conversion/main.py run --work-dir /data/gemma4-conversion --stage compile
```

LC 使用 WikiText2 train 的 128×512 校准，HPP 使用 SDK Gemma4 的 MMStar 30
样本校准；精度评估使用完整 WikiText2 test、序列长度 2048。MMStar 参与 SDK
原生校准流程，不改变本次只编译语言分支的范围。

编译输出位于 `artifacts/hpp/hbm/`。一次新的转换产生新制品后，需独立核对元数据、
文件大小和完整 SHA256，再更新模型绑定；不能用旧清单替换新字节。

本次参考工具：Python 3.10、PyTorch 2.8.0+cu128、HPP 3.3.5+cu128.torch280.qwen35、
HBDK `4.14.1a2.dev202610090350+05dd1f0.develop`。编译为 `nash-p` 4 核、
chunk 512、cache 4096、FP16 外挂 embedding。
