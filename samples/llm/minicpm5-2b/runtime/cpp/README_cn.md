> 迁移状态：进行中。以下历史板测、精度和 SDK 发布计划来自固定 S 源提交 `380e1a2`，不是本轮测试或最新发布状态。本轮只完成主机侧启动编排；量化方案保留、不重新验证，板测未运行。

[English](README.md) | [简体中文](README_cn.md)

# C++ 运行说明

<a id="supported-boards"></a>
## 支持范围

仅 S600/Nash-p；S100/S100P 走 [legacy](../legacy/README_cn.md)，不可混用模型或 SDK。源 SDK 包为 2.0.0-beta1，生成记录中的 runtime 为 2.0.4，这两个版本标签分别属于包与运行库。

<a id="dependencies"></a>
## 依赖与准备

依赖：S600、配套 OELLM 2.0.4 runtime、C++17 编译器、CMake、gflags、nlohmann-json 和 curl。在 RDK OS 执行 `sudo apt-get install build-essential cmake libgflags-dev nlohmann-json3-dev curl`。以下命令均从本目录执行。`run.sh --build` 显式构建；普通 `run.sh` 只启动已有程序并设置库路径/L2M。请独立准备模型，见[启动器参数](../README_cn.md)。如果只将 SDK 的 runtime 目录复制到了板端，可以直接设置 OELLM_RUNTIME_ROOT。

<a id="build"></a>
## 手动构建

这是启动器 `--build` 的替代方式。按下文直接使用显式 `build` 目录中的程序，或改用启动器按目标划分的构建目录。

```bash
export OELLM_SDK_ROOT=/path/to/OpenExplorer_LLM
export OELLM_RUNTIME_ROOT="$OELLM_SDK_ROOT/oellm_runtime"
cmake -S . -B build -DMINICPM_TARGET=s600 -DOELLM_RUNTIME_ROOT="$OELLM_RUNTIME_ROOT"
cmake --build build --parallel 4
export LD_LIBRARY_PATH="$OELLM_RUNTIME_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6
./build/main --model_path=../../model/s600
```

<a id="run"></a>
## 准备并运行

```bash
export OELLM_SDK_ROOT=/path/to/OpenExplorer_LLM
BOARD=s600 bash ../../model/download_model.sh
bash run.sh --build
bash run.sh
bash run.sh -- --prompt="Explain why the sky is blue in three sentences." --max_new_tokens=128
bash run.sh -- --prompt="What is the capital of France?" --follow_up="Translate that city name into Chinese."
```

<a id="parameters"></a>
## 原生参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_path` | `../../model/s600` | 已校验的模型目录；run.sh 默认路径相对脚本自身解析。 |
| `--prompt` | `What is 1+1? Give a short answer.` | 第一轮用户提示词。 |
| `--follow_up` | 空 | 共享会话状态的第二轮提示词。 |
| `--max_new_tokens` | 128 | 每轮生成上限 1–4096，总上下文仍限制为4096。 |

<a id="interface-lifecycle"></a>
## 接口与生命周期

inc/minicpm5.hpp 定义 Config、Result 和顺序调用的模型类，src/minicpm5.cc 实现初始化与推理，src/main.cc 负责 gflags 参数及输出。分词、BPU推理与解码由 runtime 完成，生成的代码文本不会被执行。初始化结束后删除临时 runtime JSON。

`MiniCPM5(Config)` 持有一个 runtime/会话，`Generate(prompt, new_chat=true)` 开始新会话，后续轮传 `false`。同一实例顺序调用，返回值拥有文本/token 数据。SDK 内部负责分词、执行和解码；本轮尚未把这些内部阶段包装成统一的公开阶段 API。
<a id="results-interpretation"></a>
## 结果解释

每次请求输出一行以 RESULT 开头的 JSON，包含 text、token_ids、status、ttft_ms、decode_tps 和 e2e_ms；同时可能出现 SDK 日志。状态3表示EOS结束，6表示达到生成上限，4表示达到上下文上限。达到限制按有界请求成功处理；输入或运行错误返回非零退出码。只生成一个token的长度限制请求可能报告 decode_tps 为0。
