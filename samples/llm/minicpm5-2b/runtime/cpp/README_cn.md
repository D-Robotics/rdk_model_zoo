> 下文的板测结果、精度与 SDK 发布说明为 S 源发布的记录；板端运行按本指南命令执行。

[English](README.md) | [简体中文](README_cn.md)

# C++ 运行说明

<a id="overview"></a>
## C++ 推理

本目录提供C++ 推理所需的程序与操作说明。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── inc/  # inc 相关文件
├── src/  # src 相关文件
├── CMakeLists.txt  # 源码或数据文件
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── run.sh  # 运行示例
```

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

inc/minicpm5.hpp 定义 Config、Result、生成阶段函数和顺序调用的模型类；src/minicpm5.cc 实现各阶段；src/runtime_config.cc 负责模型文件校验、OELLM JSON 配置与临时配置文件；src/main.cc 负责 gflags 参数及 RESULT 输出。生成的代码文本不会被执行。

公开阶段函数为 `pre_process`（构建并校验一个 OELLM 请求，不调用 SDK）、`infer`（一次同步推理调用加响应形态校验）与 `post_process`（提取文本、token 和状态，并读取与校验请求指标）；`Generate` 按此串联。分词与模板渲染保留在 runtime 内部；SDK 未暴露分词接口。配置与文件 IO 位于 `src/runtime_config.cc`，不在推理阶段文件内；临时 runtime JSON 由 RAII 守护持有，在成功、SDK 错误返回和异常路径上都会删除。

`MiniCPM5(Config)` 持有一个 runtime/会话，`Generate(prompt, new_chat=true)` 开始新会话，后续轮传 `false`。同一实例顺序调用，返回值拥有文本/token 数据。`validate_metrics` 按指标名称拒绝非有限或负数的测量值，绝不将其改写为 0，因此 RESULT 行只包含有效测量值；只生成一个 token 的长度限制请求允许 decode_tps 为 0。

完整的原生库使用示例（自包含程序）——可直接复制、对照 SDK 头文件编译并运行：

```cpp
#include "minicpm5.hpp"

int main() {
  minicpm5::Config config;              // model_path 默认 ../../model/s600
  config.max_new_tokens = 128;          // 1-4096
  minicpm5::MiniCPM5 model(config);     // 校验配置并准备 runtime；失败抛异常
  minicpm5::Result first = model.Generate("What is 1+1?");  // 开启会话
  if (first.status == 3) {              // 3 EOS；6 生成上限；4 上下文上限
    // 消费 first.text、first.tokens、first.ttft_ms、first.decode_tps、first.e2e_ms
  }
  minicpm5::Result follow = model.Generate("Translate that.", false);  // 同一会话
  return first.status == 3 && follow.status == 3 ? 0 : 1;
}
```

`pre_process` 是公开函数，因此其参数约束在函数内部强制执行：空 prompt 或超出 1–4096 的 `max_new_tokens` 对所有调用方直接抛错，与构造函数在加载模型前的校验相互独立。runtime 负责分词、BPU 执行与解码；生成的代码文本不会被执行。

<a id="results-interpretation"></a>
## 结果解释

每次请求输出一行以 RESULT 开头的 JSON，包含 text、token_ids、status、ttft_ms、decode_tps 和 e2e_ms；同时可能出现 SDK 日志。状态3表示EOS结束，6表示达到生成上限，4表示达到上下文上限。达到限制按有界请求成功处理；输入或运行错误返回非零退出码。只生成一个token的长度限制请求可能报告 decode_tps 为0。若指标非有限或为负数，该请求以非零退出码失败，不会输出 RESULT 行。
