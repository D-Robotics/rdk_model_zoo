[English](README.md) | [简体中文](README_cn.md)

# C++ 运行说明

依赖：S600、配套 OELLM 2.0.4 runtime、C++17 编译器、CMake、gflags、nlohmann-json 和 curl。在 RDK OS 执行 `sudo apt-get install build-essential cmake libgflags-dev nlohmann-json3-dev curl`。以下命令均从本目录执行。run.sh 会校验或下载模型、编译程序并设置库路径和 L2M 环境。如果只将 SDK 的 runtime 目录复制到了板端，可以直接设置 OELLM_RUNTIME_ROOT。

```bash
export OELLM_SDK_ROOT=/path/to/OpenExplorer_LLM
bash run.sh
bash run.sh --prompt="Explain why the sky is blue in three sentences." --max_new_tokens=128
bash run.sh --prompt="What is the capital of France?" --follow_up="Translate that city name into Chinese."
```

```bash
export OELLM_RUNTIME_ROOT="$OELLM_SDK_ROOT/oellm_runtime"
cmake -S . -B build -DOELLM_RUNTIME_ROOT="$OELLM_RUNTIME_ROOT"
cmake --build build --parallel 4
export LD_LIBRARY_PATH="$OELLM_RUNTIME_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6
./build/main --model_path=../../model/s600
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_path` | `../../model/s600` | 已校验的模型目录；run.sh 默认路径相对脚本自身解析。 |
| `--prompt` | `What is 1+1? Give a short answer.` | 第一轮用户提示词。 |
| `--follow_up` | 空 | 共享会话状态的第二轮提示词。 |
| `--max_new_tokens` | 128 | 每轮生成上限 1–4096，总上下文仍限制为4096。 |

每次请求输出一行以 RESULT 开头的 JSON，包含 text、token_ids、status、ttft_ms、decode_tps 和 e2e_ms；同时可能出现 SDK 日志。状态3表示EOS结束，6表示达到生成上限，4表示达到上下文上限。达到限制按有界请求成功处理；输入或运行错误返回非零退出码。只生成一个token的长度限制请求可能报告 decode_tps 为0。

inc/minicpm5.hpp 定义 Config、Result 和顺序调用的模型类，src/minicpm5.cc 实现初始化与推理，src/main.cc 负责 gflags 参数及输出。分词、BPU推理与解码由 runtime 完成，生成的代码文本不会被执行。初始化结束后删除临时 runtime JSON。
