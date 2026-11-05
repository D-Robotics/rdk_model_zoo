[English](README.md) | 简体中文

# 官方 C++ Runtime 入口

<a id="overview"></a>
## 推理入口

LLM/VLM 直接运行 SDK `samples/basic/basic_sample`；ASR 直接运行 SDK `examples/asr_demo/asr_demo`。tokenizer、embedding、视觉/音频处理、prefill/decode、KV 与投机解码全部由官方 Runtime 实现。

<a id="directory"></a>
## 目录

`run.sh` 加载 SDK 环境；`run.py` 校验准备好的配置并调用原生程序；`configs/` 保存每个配置的 SDK 原文。

<a id="supported-boards"></a>
## 支持平台

S600 / Linux / aarch64 / nash-p. 启动时检查真实板型。`--dry-run` 可在主机查看命令。

<a id="dependencies"></a>
## 依赖

完整 SDK 2.0.0 S600 Runtime、Python 3 标准库与 Bash。SDK 原生程序所需动态库由对应部署包提供；ASR 音频解码使用 SDK 的 FFmpeg 链路。

<a id="build"></a>
## 构建

缺省使用 SDK 已提供的可执行程序。需要重编译时添加 `--build`，启动器从 SDK 对应目录的 CMake 项目构建到本 Sample 的 `runtime/cpp/build/`。C++ 源码、头文件与库仍从用户准备的 SDK 取得。

<a id="run"></a>
## 运行

从仓库根目录运行。先为每个示例准备其对应官方包。

```bash
export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600
# Text
bash samples/llm/oellm/runtime/cpp/run.sh --config_path /data/Qwen3-0.6B_w8/oellm_config.json --prompt "你好"
# Image + text (prepare qwen3-vl-2b-w4 first)
bash samples/llm/oellm/runtime/cpp/run.sh --config_path /data/Qwen3-VL-2B_w4/oellm_config.json \
  --image_path "$OELLM_RUNTIME_ROOT/resource/image.jpg" --prompt "描述图片"
# ASR (prepare qwen3-asr-0-6b-w8 first)
bash samples/llm/oellm/runtime/cpp/run.sh --config_path /data/Qwen3-ASR-0.6B_w8/oellm_config.json \
  --audio_path "$OELLM_RUNTIME_ROOT/resource/test_asr.flac"
```

<a id="parameters"></a>
## 参数

| Option | Default | 作用 |
| --- | --- | --- |
| `--config_path` | 必填 | 准备好的 JSON |
| `--prompt` | `你好` | 文本/图文提示 |
| `--image_path` | 空 | VLM image |
| `--audio_path` | 空 | ASR audio |
| `--sampler_config_path` | 空 | SDK basic sampler JSON |
| `--build` | false | 从 SDK 自带源码构建 |
| `--dry-run` | false | 仅输出命令 |

<a id="interface-lifecycle"></a>
## 请求与生命周期

入口一次启动处理一个请求。文本/图文固定使用 SDK basic 的同步模式并跳过预热；ASR 使用整文件转写。加速模型保留官方 `infer_acceleration` 配置。`--sampler_config_path` 仅传给 basic 程序。ASR 默认语言来自配置的 `asr_param.language`，需要其他语言时按官方配置约定设置。

<a id="results-interpretation"></a>
## 结果解读

输出保留 SDK 原生格式。SDK 请求指标与网页官方参考的输入条件可能不同，比较时以 [评估说明](../../evaluator/README_cn.md) 为准。本项目板端编译、加载、生成和性能测试均为 `not-run`。
