# SDK 2.0 C++ 文本入口

[English](README.md) · [Sample](../../README_cn.md)

需要 S600、配套的 SDK 2.0.0 `oellm_runtime_s600`、CMake 和 C++17 编译器。
JSON 头文件使用 SDK 自带 `3rdparty/nlohmann`；只链接 SDK 官方
`liboellm_runtime.so`，不携带旧版自定义 Gemma4 引擎。

先按 [模型准备](../../model/README_cn.md) 生成绝对路径配置，再执行：

```bash
export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600
bash samples/llm/gemma4/runtime/cpp/run.sh \
  --config_path /data/gemma4-e2b-sdk2/oellm_config.json \
  --prompt "What is 2 + 2? Answer with the number only." \
  --follow_up "请用中文写出这个数字。" --max_new_tokens 64
```

也可设置 `OELLM_SDK_ROOT`，脚本从其 `oellm_runtime_s600` 目录取运行库。
`run.sh` source 官方 `env.sh` 后编译并启动。手动构建：

```bash
cmake -S samples/llm/gemma4/runtime/cpp -B /tmp/gemma4-build \
  -DOELLM_RUNTIME_ROOT="$OELLM_RUNTIME_ROOT"
cmake --build /tmp/gemma4-build --parallel 4
source "$OELLM_RUNTIME_ROOT/env.sh" 4
/tmp/gemma4-build/main --config_path /data/gemma4-e2b-sdk2/oellm_config.json --prompt "你好"
```

| 参数 | 说明 |
| --- | --- |
| `--config_path` | 必填，`prepare.py` 生成的配置 |
| `--prompt` | 单次文本提问；默认简单算术题 |
| `--follow_up` | 可选，携带完整上一轮历史追问 |
| `--max_new_tokens` | 默认 128，允许 1–4096；仍受总上下文容量限制 |
| `--interactive` | 终端文本对话，支持 `/reset` 与 `/quit` |

每次生成输出一行 `RESULT {JSON}`，包含文本、token、停止状态及 SDK 可用的请求指标。
SDK 未返回的指标为 null。初始化、推理失败返回非零退出码。
本入口不接收图像、音频或视频；配置中的视觉 HBM 是 SDK 初始化依赖。

SDK 2.0 每轮需要完整 `prompt.history`，`conversation_id` 不代替对话历史。
此实现显式保留 user/assistant 历史，贪心生成，默认不启用 prefix cache。
主表与 PLE 表沿用本次编译文件，不套用旧 SDK 包的缩放补偿。

板端编译、加载、生成和性能测试当前均为 `not-run`。
