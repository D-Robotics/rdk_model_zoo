> Migration status: in progress. Board results, accuracy and SDK release forecasts below are historical records from pinned S source `380e1a2`, not new tests or current release status. This round covers host launch orchestration only; quantization recipes are preserved without rerunning, and board tests are not-run.

[English](README.md) | [简体中文](README_cn.md)

# C++ runtime

<a id="supported-boards"></a>
## Supported boards

S600/Nash-p only; S100/S100P use [legacy](../legacy/README.md). Models and SDKs are not interchangeable. The source SDK package is 2.0.0-beta1; generation records name runtime 2.0.4, a separate runtime version label.

<a id="dependencies"></a>
## Dependencies and preparation

Requirements: S600, the supplied OELLM 2.0.4 runtime, C++17 compiler, CMake, gflags, nlohmann-json and curl. On RDK OS use `sudo apt-get install build-essential cmake libgflags-dev nlohmann-json3-dev curl`. Run commands from this directory. `run.sh --build` explicitly builds; ordinary `run.sh` only launches the existing binary and sets library/L2M variables. Prepare the model separately. See [launcher options](../README.md). Set OELLM_RUNTIME_ROOT instead if only the SDK runtime directory was copied to the board.

<a id="build"></a>
## Manual build

This is an alternative to launcher `--build`. Use its explicit `build` binary below, or use the launcher’s separate target-specific build directory.

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
## Prepare and run

```bash
export OELLM_SDK_ROOT=/path/to/OpenExplorer_LLM
BOARD=s600 bash ../../model/download_model.sh
bash run.sh --build
bash run.sh
bash run.sh -- --prompt="Explain why the sky is blue in three sentences." --max_new_tokens=128
bash run.sh -- --prompt="What is the capital of France?" --follow_up="Translate that city name into Chinese."
```

<a id="parameters"></a>
## Native parameters

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model_path` | `../../model/s600` | Verified model directory; run.sh resolves its default relative to itself. |
| `--prompt` | `What is 1+1? Give a short answer.` | First user prompt. |
| `--follow_up` | empty | Optional second prompt sharing conversation state. |
| `--max_new_tokens` | 128 | Output limit per request, 1–4096; total context remains 4096. |

<a id="interface-lifecycle"></a>
## Interface and lifecycle

`inc/minicpm5.hpp` defines Config, Result and the sequential MiniCPM5 wrapper; `src/minicpm5.cc` implements model setup and generation; `src/main.cc` handles gflags and output. The runtime performs tokenization, BPU execution and decoding. No generated text is executed as code. The temporary runtime JSON is removed after initialization.

`MiniCPM5(Config)` owns one runtime/conversation. `Generate(prompt, new_chat=true)` starts a new conversation; pass `false` for a follow-up. Calls on one instance are sequential; the returned value owns text/token data. Tokenization, execution and decoding are SDK-owned; a unified public stage API has not yet been introduced in this migration.
<a id="results-interpretation"></a>
## Interpret results

Each request prints a `RESULT` JSON line with text, token_ids, status, ttft_ms, decode_tps and e2e_ms. SDK diagnostics may also appear. Status 3 means EOS, 6 means the output limit and 4 means the context limit. Limits are reported as successful bounded requests; invalid input or runtime errors return nonzero. A one-token length-limited request may report zero decode_tps.
