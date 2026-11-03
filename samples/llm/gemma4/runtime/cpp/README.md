# SDK 2.0 C++ text entry point

[简体中文](README_cn.md) · [Sample](../../README.md)

Requires S600, the matching SDK 2.0.0 `oellm_runtime_s600`, CMake and a C++17
compiler. JSON headers come from SDK `3rdparty/nlohmann`. The application links
the official `liboellm_runtime.so` and delegates the model engine to its basic API.

After [preparation](../../model/README.md), from the repository root:

```bash
export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600
bash samples/llm/gemma4/runtime/cpp/run.sh \
  --config_path /data/gemma4-e2b-sdk2/oellm_config.json \
  --prompt "What is 2 + 2? Answer with the number only." \
  --follow_up "Write that number in Chinese." --max_new_tokens 64
```

Alternatively set `OELLM_SDK_ROOT`; its `oellm_runtime_s600` subdirectory is used.
The script sources official `env.sh`, builds, and starts the program. For manual
builds set CMake `OELLM_RUNTIME_ROOT` and source `env.sh` with argument `4` before
launching the resulting `main` executable.

| Option | Meaning |
| --- | --- |
| `--config_path` | Required prepared runtime JSON |
| `--prompt` | Text prompt; defaults to a short arithmetic question |
| `--follow_up` | Optional second request with full previous history |
| `--max_new_tokens` | Default 128, range 1–4096; total context is still bounded |
| `--interactive` | Text shell; `/reset` clears history and `/quit` exits |

Each request emits `RESULT {JSON}` with text, token IDs, stop status and metrics
returned by the SDK. Unavailable metrics are null. Initialization and inference
failures exit nonzero. The CLI accepts no image, audio or video input; its vision
HBM is required by SDK initialization.

SDK 2.0 expects complete `prompt.history` per request. This wrapper retains
user/assistant history and uses greedy decoding. Prefix reuse is disabled by
default. Embeddings retain this conversion's bytes without old-release scaling
compensation. Board build, loading, generation and performance tests are `not-run`.
