[English](README.md) | 简体中文

# 官方 Runtime 文本入口

[Sample](../../README_cn.md) · [模型准备](../../model/README_cn.md)

<a id="overview"></a>
## 文本推理

调用官方 basic API 进行单次文本生成、完整历史的多轮生成及终端对话。

<a id="directory"></a>
## 目录

`src/main.cc` 提供 CLI，`src/gemma4.cc` 与 `inc/gemma4.hpp` 封装 Runtime，`configs/` 保存四个配置，`CMakeLists.txt` 与 `run.sh` 负责构建启动。

<a id="supported-boards"></a>
## 支持板卡

S600/Linux aarch64, nash-p, 4 BPU cores, context 4096, prefill 512.

<a id="dependencies"></a>
## 依赖

支持四个官方 S600 配置。需要配套 SDK 2.0.0 `oellm_runtime_s600`、CMake/C++17；JSON 头文件使用 SDK 自带版本。


<a id="build"></a>
## 构建

`run.sh` 使用 CMake 3.16+ 在 S600 构建 C++17 程序，加载 SDK `env.sh 4` 并链接 `liboellm_runtime.so`。

<a id="run"></a>
## 运行

完成模型准备后，从仓库根目录运行：

```bash
export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600
bash samples/llm/gemma4/runtime/cpp/run.sh \
  --config_path /data/gemma4-official/Gemma4-E2B_w4/oellm_config.json \
  --prompt "What is 2 + 2? Answer with the number only." \
  --follow_up "请用中文写出这个数字。" --max_new_tokens 64
```

将配置路径换为相应 E2B/E4B W4/W8 目录即可切换模型。`run.sh` 加载 SDK `env.sh`，只链接官方 `liboellm_runtime.so`。
`--interactive` 启用终端对话；`/reset` 清空历史，`/quit` 退出。
`--max_new_tokens` 默认 128，取值 1–4096，仍受总上下文容量限制。

<a id="parameters"></a>
## 参数

| Option | Default | 说明 |
| --- | --- | --- |
| `--config_path` | 必填 | 生成的 Runtime JSON |
| `--prompt` | `What is 2 + 2? Answer with the number only.` | 首轮提示 |
| `--follow_up` | 空 | 可选第二轮提示 |
| `--max_new_tokens` | `128` | 1–4096 |
| `--interactive` | `false` | 终端对话 |

<a id="interface-lifecycle"></a>
## 接口与生命周期

SDK 每轮需要完整 `prompt.history`。本入口显式保留 user/assistant 历史并使用贪心生成。
前缀缓存等 Runtime 设置沿用官方配置；每次输出 `RESULT {JSON}`，包含文本、token、状态和 SDK 可用指标。
SDK 没有返回的指标保持 null。请求接口仅接受文本，官方模型文件保留完整。

<a id="results-interpretation"></a>
## 结果解读

`RESULT` 中的指标只表示该次 SDK 请求；官方参考数据另见 [evaluator](../../evaluator/README_cn.md)。

本轮未运行板端编译、加载、生成或性能测试。
