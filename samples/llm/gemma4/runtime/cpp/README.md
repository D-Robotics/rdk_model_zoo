English | [简体中文](README_cn.md)

# Official Runtime text entrypoint

[Sample](../../README.md) · [Model preparation](../../model/README.md)

<a id="overview"></a>
## Text inference

Use the official basic API for single prompts, full-history follow-ups and terminal chat.

<a id="directory"></a>
## Directory

`src/main.cc` provides the CLI; `src/gemma4.cc` and `inc/gemma4.hpp` wrap the Runtime, `configs/` holds four profiles, and `CMakeLists.txt`/`run.sh` build and launch the client.

<a id="supported-boards"></a>
## Supported board

S600/Linux aarch64, nash-p, 4 BPU cores, context 4096, prefill 512.

<a id="dependencies"></a>
## Dependencies

All four official profiles use the matching SDK 2.0.0 `oellm_runtime_s600`, CMake/C++17, and SDK-provided JSON headers.


<a id="build"></a>
## Build

`run.sh` builds the C++17 program with CMake 3.16+ on S600, sources SDK `env.sh 4` and links `liboellm_runtime.so`.

<a id="run"></a>
## Run

After preparing the model, run from the repository root:

```bash
export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600
bash samples/llm/gemma4/runtime/cpp/run.sh \
  --config_path /data/gemma4-official/Gemma4-E2B_w4/oellm_config.json \
  --prompt "What is 2 + 2? Answer with the number only." \
  --follow_up "Write this number in Chinese." --max_new_tokens 64
```

Choose the matching E2B/E4B W4/W8 configuration path. `run.sh` loads SDK `env.sh` and links the official `liboellm_runtime.so`.
Use `--interactive` for terminal chat, `/reset` to clear history, and `/quit` to exit.
`--max_new_tokens` defaults to 128, accepts 1–4096, and remains bounded by the total context capacity.

<a id="parameters"></a>
## Parameters

| Option | Default | Purpose |
| --- | --- | --- |
| `--config_path` | required | Prepared Runtime JSON |
| `--prompt` | `What is 2 + 2? Answer with the number only.` | First prompt |
| `--follow_up` | empty | Optional second prompt |
| `--max_new_tokens` | `128` | 1–4096 |
| `--interactive` | `false` | Terminal chat |

<a id="interface-lifecycle"></a>
## Interface and lifecycle

The SDK requires full `prompt.history` on each request. The client retains user/assistant history and uses greedy generation.
Prefix caching and other Runtime settings follow the vendor configuration.
Each generation emits `RESULT {JSON}` with text, tokens, status and available SDK metrics; unavailable metrics remain null.
The request interface accepts text while retaining the complete official model files.

<a id="results-interpretation"></a>
## Interpret results

`RESULT` metrics describe that SDK request. Published references are separate in [evaluator](../../evaluator/README.md).

No on-device build, load, generation or performance test was run for this publication.
