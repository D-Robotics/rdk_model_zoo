[English](README.md) | 简体中文

# Gemma4 E2B / E4B 文本生成

[English](README.md) · [模型准备](model/README_cn.md) · [模型来源](conversion/README_cn.md) · [运行](runtime/cpp/README_cn.md) · [性能来源](evaluator/README_cn.md)

<a id="overview"></a>
## 模型与来源

本 Sample 使用 OELLM SDK 2.0.0 官方 S600 模型包及 `OellmRuntime` basic API。
提供 E2B/E4B 的 W4、W8 四个配置，支持单次提问、完整历史的多轮对话和终端交互。
分词、普通/PLE embedding、prefill/decode、KV cache 和采样均由官方 Runtime 实现。

<a id="directory"></a>
## 目录

```text
gemma4/
├── model/          # 四个官方配置、校验及准备
├── runtime/cpp/    # OellmRuntime basic API
├── conversion/     # 官方制品来源
├── evaluator/      # 官方基准及独立板端记录
├── test_data/      # 文本提示
└── tests/          # 主机文件校验测试
```

<a id="support-matrix"></a>
## 支持的模型与板卡

下列配置均使用 S600/Linux 的 C++ 入口。

| Profile | ZIP | Bytes |
| --- | --- | ---: |
| `e2b-w4` | [Gemma4-E2B_w4.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E2B_w4.zip) | 4,402,995,725 |
| `e2b-w8` | [Gemma4-E2B_w8.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E2B_w8.zip) | 6,547,749,326 |
| `e4b-w4` | [Gemma4-E4B_w4.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E4B_w4.zip) | 6,320,670,645 |
| `e4b-w8` | [Gemma4-E4B_w8.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E4B_w8.zip) | 9,762,116,686 |

四个配置均为 S600 `nash-p`、4 核、chunk 512、上下文 4096；其他板卡不使用这些包。
W4/W8 是官方语言主干的精度标注，视觉和音频 HBM 为 W8，外挂 embedding 为 FP16。

<a id="prerequisites"></a>
## 环境准备

在 S600 上准备配套 `oellm_runtime_s600`、CMake/C++17 及 Python 3 标准库。
需保留 ZIP 与解压后文件的空间；各包尺寸见上表。

<a id="quickstart"></a>
## 快速体验

从仓库根目录运行，下面以 E2B W4 为例：

```bash
mkdir -p /data/gemma4-official
curl -fL --retry 3 -o Gemma4-E2B_w4.zip https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E2B_w4.zip
printf '%s  %s\n' 'bd9b1406e2cf4a6cde7cda769fbcbbf43d1fa9a797f59936def6f7d6dbf793d2' 'Gemma4-E2B_w4.zip' | sha256sum -c -
python3 -m zipfile -e Gemma4-E2B_w4.zip /data/gemma4-official
python3 samples/llm/gemma4/model/prepare.py \
  --profile e2b-w4 \
  --source-dir /data/gemma4-official/Gemma4-E2B_w4 \
  --output-dir /data/gemma4-official/Gemma4-E2B_w4

export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600
bash samples/llm/gemma4/runtime/cpp/run.sh \
  --config_path /data/gemma4-official/Gemma4-E2B_w4/oellm_config.json \
  --prompt "请把 hello 翻译成中文，只回答译文。" --max_new_tokens 64
```

其他配置使用表中的 ZIP，并将 `--profile` 改为 `e2b-w8`、`e4b-w4` 或 `e4b-w8`。
各包 SHA256 和文件校验值在 `model/manifests/<profile>.json`。
终端对话使用 `--interactive`，`/reset` 清空历史，`/quit` 退出。

<a id="expected-results"></a>
## 输出与性能参考

成功时输出 `RESULT {JSON}`，含生成文本、token IDs、结束状态与可用 Runtime 指标。



以下数字来自 [SDK 官方 S600 基准](https://developer.d-robotics.cc/oe_llm_doc/guide/benchmark/s600/gemma_benchmark.html)，使用纯文本 TTFT 列：

| Profile | TTFT (ms) | Decode (token/s) | Memory (GB) |
| --- | ---: | ---: | ---: |
| e2b-w4 | 74 | 61.2 | 9.9 |
| e2b-w8 | 79 | 47.8 | 13.5 |
| e4b-w4 | 137 | 38.3 | 15.4 |
| e4b-w8 | 139 | 27.9 | 15.5 |

条件为 S600/Linux、4 核、chunk 512、上下文 4096、标准单请求模式。
内存为官方完整模型的 Linux+ION 总占用，不是纯文本峰值内存。
原始页面未公布对应包的二进制 SHA256 绑定、PPL 或两路并发数据。
本地板端测试保持 `not-run`，数值保存在独立的 `evaluator/board_results/` 空记录中。

<a id="entry-points"></a>
## 入口与主机检查

[模型准备](model/README_cn.md)、[推理入口](runtime/cpp/README_cn.md)、[制品来源](conversion/README_cn.md)、[性能与评估](evaluator/README_cn.md)。

```bash
python3 -m unittest discover -s samples/llm/gemma4/tests
```

模型准备和推理不依赖 Skill、Node 或网页生成目录，也不自动安装依赖。
现有 `develop` 的 `samples/llm/gemma4-e2b` 是另一套原生 C++ Sample；本目录复用 SDK 官方 Runtime。

<a id="license"></a>
## 来源与许可

官方 ZIP 的完整 SHA256、文件校验值、真实编译器和 Runtime 配置来源已登记。
SDK 未在 ZIP 中提供上游 HF 完整 revision，因此记录为 null，未借用其他转换的 revision。
模型采用 [Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license)，Sample 代码采用 [Apache-2.0](LICENSE)。
SDK 运行库和头文件遵循其随附条款，需单独准备。
