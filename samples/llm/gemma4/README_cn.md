# Gemma4-E2B 纯文本生成

[English](README.md) · [模型准备](model/README_cn.md) · [转换](conversion/README_cn.md) · [运行](runtime/cpp/README_cn.md) · [评测](evaluator/README_cn.md)

本 Sample 使用 **OELLM SDK 2.0.0 官方 `OellmRuntime`**，在 S600 上接入
Gemma4-E2B 文本生成。支持单次提问、完整历史的多轮对话和终端交互。
分词、Gemma4 PLE embedding、prefill/decode、KV cache 和采样均由 SDK 实现。

| 项目 | 当前记录 |
| --- | --- |
| 原始权重 | `google/gemma-4-E2B-it-qat-q4_0-unquantized` |
| revision | `6befbaca7398925921802abd1f277b495b78b738` |
| 转换 | SDK AutoClip → GPTQ → HPP → HBDK；已生成 S600 HBM |
| 精度 | W4/W8 混合：Q/O/MLP 等 W4，K/V 投影 W8；SDK A8 策略 |
| 编译规格 | `nash-p`，4 核，prefill chunk 512，总上下文 4096 |
| 板端精度 / 性能 / 内存 | `not-run`；数值为空 |
| 制品 | 完整 Archive 权重包，文件大小与 SHA256 已登记 |
| 公共模型下载 | [S600 权重，4.76 GB](https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/rdk_s600/gemma/gemma4-e2b-w4w8-s600-chunk512-ctx4096.tar.gz) |

当前只提供 S600 配置。X5、S100、S100P 没有本版本的对应制品。
旧版 `platforms/s/samples/llm/gemma4-e2b` 的 OE-LLM 1.0 模型和 runtime
保留独立版本；本 Sample 使用自己的 SDK 2.0 模型绑定。

## 快速使用

先按 [模型准备](model/README_cn.md) 下载并解压完整 Archive 权重包，再从仓库根目录执行：

```bash
python3 samples/llm/gemma4/model/prepare.py \
  --source-dir /data/gemma4-e2b-sdk2 \
  --output-dir /data/gemma4-e2b-sdk2

# 以下命令在 S600 上执行，需要配套 SDK 2.0 部署包及 CMake/C++17 工具。
export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600
bash samples/llm/gemma4/runtime/cpp/run.sh \
  --config_path /data/gemma4-e2b-sdk2/oellm_config.json \
  --prompt "请把 hello 翻译成中文，只回答译文。" --max_new_tokens 64
```

SDK 2.0.0 的 `GEMMA4` runtime 初始化时要求视觉 HBM，因此准备步骤还会校验
官方视觉配套文件。本入口只接受文本，不提供图像、音频或视频参数。
模型包约 7.30 GB；这是文件大小，不是板端运行内存测量。

终端对话：

```bash
bash samples/llm/gemma4/runtime/cpp/run.sh \
  --config_path /data/gemma4-e2b-sdk2/oellm_config.json --interactive
```

`/reset` 清空客户端历史，`/quit` 退出。每轮将完整历史交给 SDK；总上下文包含
模板、历史、当前问题和生成结果，受 4096 token 容量限制。

## 源码与检查

`model/artifacts.json` 绑定权重来源、编译器、目标规格及全部文件校验值；
`model/prepare.py` 只做显式文件准备；`runtime/cpp/s600.json` 从 SDK 官方
Gemma4 配置裁出文本入口所需配置；`gemma4.cc` 调用 SDK basic API。
推理不依赖 Skill、Node 或网页生成目录。运行脚本不安装依赖或下载模型。

```bash
python3 -m unittest discover -s samples/llm/gemma4/tests
```

主机量化参考结果单列在 [评测说明](evaluator/README_cn.md)。板端结果模板为
`evaluator/board_result.json`，保持 `not-run` 与 `null`。

## 许可证

模型采用 [Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license)，上游模型卡
随部署制品保留。Sample 代码采用 [Apache-2.0](LICENSE)，与 S 系列示例许可一致。
SDK 运行库、头文件使用其各自条款，需单独获取；本仓库不打包这些库。
权重包中的官方配套模型保留独立的 SDK 来源与条款。
