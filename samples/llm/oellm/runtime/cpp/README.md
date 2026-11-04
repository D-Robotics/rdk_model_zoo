English | [简体中文](README_cn.md)

# Official C++ Runtime entrypoints

<a id="overview"></a>
## Inference entrypoints

LLM/VLM directly run SDK `samples/basic/basic_sample`; ASR directly runs SDK `examples/asr_demo/asr_demo`. The official Runtime owns tokenization, embeddings, vision/audio processing, prefill/decode, KV state and speculative decoding.

<a id="directory"></a>
## Directory

`run.sh` loads the SDK environment; `run.py` checks prepared configuration and launches the native program; `configs/` holds original SDK configurations.

<a id="supported-boards"></a>
## Supported platform

S600 / Linux / aarch64 / nash-p. Execution checks actual board identity. Host `--dry-run` previews the command.

<a id="dependencies"></a>
## Dependencies

Complete SDK 2.0.0 S600 Runtime, Python 3 standard library and Bash. The deployment package supplies native dependencies; ASR uses its FFmpeg decoding path.

<a id="build"></a>
## Build

Use vendor-supplied executables by default. Optional `--build` builds their SDK CMake projects into this sample’s `runtime/cpp/build/`. C++ source, headers and libraries remain in the supplied SDK.

<a id="run"></a>
## Run

Run from the repository root after preparing the matching official package for each example.

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
## Parameters

| Option | Default | Purpose |
| --- | --- | --- |
| `--config_path` | required | Prepared JSON |
| `--prompt` | `你好` | Text/image prompt |
| `--image_path` | empty | VLM image |
| `--audio_path` | empty | ASR audio |
| `--sampler_config_path` | empty | SDK basic sampler JSON |
| `--build` | false | Build supplied SDK source |
| `--dry-run` | false | Print command only |

<a id="interface-lifecycle"></a>
## Requests and lifecycle

Each invocation processes one request. Text/images use synchronous SDK basic mode without warmup; ASR uses whole-file transcription. Accelerated profiles retain vendor `infer_acceleration`. Sampler configuration is forwarded only to basic. ASR language comes from `asr_param.language`; use the vendor configuration contract when selecting another language.

<a id="results-interpretation"></a>
## Interpret results

Output retains SDK native format. Request metrics and published references may use different inputs; see [evaluation](../../evaluator/README.md). Board build, load, generation and performance tests for this integration are `not-run`.
