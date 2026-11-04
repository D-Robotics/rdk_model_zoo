[English](README.md) | 简体中文

# OELLM 官方 S600 模型

<a id="overview"></a>
## 模型与来源

使用 OELLM SDK 2.0.0 官方 S600 ZIP 和配套 Runtime，覆盖 32 个文本、图文、语音配置及 Eagle3/DFlash 加速变体。模型下载直接使用 SDK 官方 OSS；现有 Gemma 4 位于 [独立 Sample](../gemma4/README_cn.md)。

<a id="directory"></a>
## 目录

```text
oellm/
├── model/          # 官方清单、校验和准备
├── runtime/cpp/    # 官方 C++ 程序启动器与配置
├── conversion/     # 预编译制品来源
├── evaluator/      # 官方基准与独立板端记录
├── test_data/      # 文本与 SDK 输入路径示例
└── tests/          # 主机检查
```

<a id="support-matrix"></a>
## 支持的模型与平台

| 系列 | 配置数量 | 任务 |
| --- | ---: | --- |
| InternVL2 | 1 | vlm |
| InternVL2.5 | 1 | vlm |
| InternVL3.5 | 1 | vlm |
| MiniCPM5 | 1 | text-gen |
| Phi-4 Mini | 1 | text-gen |
| Qwen2.5 | 1 | text-gen |
| Qwen2.5-VL | 2 | vlm |
| Qwen3 | 11 | text-gen |
| Qwen3.5 | 3 | vlm |
| Qwen3-ASR | 2 | asr |
| Qwen3-VL | 6 | vlm |
| Whisper | 2 | asr |

全部配置面向 S600/Linux aarch64。常规模型与加速模型使用各自的完整 ZIP 和配置，不混用；Qwen3.5 4B 的语言/视觉图为 1 核，其余语言图为 4 核，DFlash 草稿图单独使用 1 核。

<a id="prerequisites"></a>
## 环境准备

准备完整的 SDK 2.0.0 `oellm_runtime_s600`，包含 `env.sh`、`lib/`、`include/`、`samples/basic/basic_sample`、`examples/asr_demo/asr_demo` 和 `resource/`。启动器使用 Python 3 标准库和 Bash；若选择 `--build`，还需 CMake 3.15+ 与 C++17。ZIP 与解压文件的磁盘需求见 [模型清单](model/README_cn.md)。

<a id="quickstart"></a>
## 快速体验

从仓库根目录在 S600 上运行，以 Qwen3 0.6B W8 为例：

```bash
curl -fL --retry 3 -o Qwen3-0.6B_w8.zip https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Qwen3-0.6B_w8.zip
printf '%s  %s\n' '54d95f45185d61ae284d270f1341f4d533dc68ce4263c78067f14c0ca5725327' 'Qwen3-0.6B_w8.zip' | sha256sum -c -
python3 -m zipfile -e Qwen3-0.6B_w8.zip /data
MODEL_DIR=/data/Qwen3-0.6B_w8
python3 samples/llm/oellm/model/prepare.py --profile qwen3-0-6b-w8 --source-dir "$MODEL_DIR"
export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600
bash samples/llm/oellm/runtime/cpp/run.sh --config_path "$MODEL_DIR/oellm_config.json" --prompt "你好"
```

列出其他配置：

```bash
python3 samples/llm/oellm/model/prepare.py --list
```

其他配置下载其对应 ZIP，修改 `--profile` 和解压目录。图文和语音命令见 [Runtime](runtime/cpp/README_cn.md#run)。

<a id="expected-results"></a>
## 输出与性能

文本/图文入口由 SDK `basic_sample` 输出 `text_result` 和 Runtime 指标；ASR 由 SDK `asr_demo` 输出 `[Transcription]` 与 `[Metric]`。页面数字来自 SDK 官方基准，速度范围保持原样；这些参考与本项目的板端记录分开保存。

<a id="entry-points"></a>
## 入口

[model](model/README_cn.md) · [runtime](runtime/cpp/README_cn.md) · [conversion](conversion/README_cn.md) · [evaluator](evaluator/README_cn.md)

```bash
python3 -m unittest discover -s samples/llm/oellm/tests
```

模型准备和原生推理不依赖 Skill、Node 或网页目录，不自动安装组件。

<a id="license"></a>
## 来源与许可

Sample 脚本采用 [Apache-2.0](LICENSE)。权重许可按 `model/manifests/<profile>.json` 的 `source.license` 保存；Qwen2.5 3B 与 Qwen2.5-VL 3B 为 Qwen RESEARCH LICENSE AGREEMENT，Phi-4 Mini、InternVL2/2.5 为 MIT，其余此批基础模型为 Apache-2.0。上游 HF 完整 revision 未由官方包披露，保持 null。SDK C++ 程序、头文件、运行库遵循随附 SDK 条款，需单独准备。
