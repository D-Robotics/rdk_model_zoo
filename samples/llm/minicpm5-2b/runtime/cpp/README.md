> Board results, accuracy figures and SDK release notes below are records from the source S release; board runs follow the commands in this guide.

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

`inc/minicpm5.hpp` defines Config, Result, the generation-stage functions and the sequential MiniCPM5 wrapper; `src/minicpm5.cc` implements the stages; `src/runtime_config.cc` owns model-file validation, the OELLM JSON settings and the temporary configuration file; `src/main.cc` handles gflags and RESULT output. No generated text is executed as code.

The public stage functions are `pre_process` (builds and validates one OELLM request; no SDK calls), `infer` (one synchronous runtime call plus a response-shape check) and `post_process` (extracts text, tokens and status, then reads and validates request metrics); `Generate` chains them. Tokenization and template rendering stay inside the runtime; the SDK exposes no tokenization API. Configuration and file IO live in `src/runtime_config.cc`, outside the inference-stage file; the temporary runtime JSON is owned by an RAII guard and removed on success, SDK error return and exception paths alike.

`MiniCPM5(Config)` owns one runtime/conversation. `Generate(prompt, new_chat=true)` starts a new conversation; pass `false` for a follow-up. Calls on one instance are sequential; the returned value owns text/token data. `validate_metrics` rejects non-finite or negative measurements with a message naming the metric and never coerces them to zero, so a RESULT line only contains valid measurements; zero `decode_tps` remains valid for one-token length-limited requests.

Complete native library usage as one self-contained program — copy it, compile against the SDK headers and run:

```cpp
#include "minicpm5.hpp"

int main() {
  minicpm5::Config config;              // model_path defaults to ../../model/s600
  config.max_new_tokens = 128;          // 1-4096
  minicpm5::MiniCPM5 model(config);     // validates settings, prepares the runtime; throws on failure
  minicpm5::Result first = model.Generate("What is 1+1?");  // opens the conversation
  if (first.status == 3) {              // 3 EOS; 6 output limit; 4 context limit
    // consume first.text, first.tokens, first.ttft_ms, first.decode_tps, first.e2e_ms
  }
  minicpm5::Result follow = model.Generate("Translate that.", false);  // same conversation
  return first.status == 3 && follow.status == 3 ? 0 : 1;
}
```

`pre_process` is public, so its own argument contract is enforced inside it: an empty prompt or a `max_new_tokens` outside 1–4096 throws there for every caller, independent of the constructor check that runs before model loading. The runtime performs tokenization, BPU execution and decoding; no generated text is executed as code.
<a id="results-interpretation"></a>
## Interpret results

Each request prints a `RESULT` JSON line with text, token_ids, status, ttft_ms, decode_tps and e2e_ms. SDK diagnostics may also appear. Status 3 means EOS, 6 means the output limit and 4 means the context limit. Limits are reported as successful bounded requests; invalid input or runtime errors return nonzero. A one-token length-limited request may report zero decode_tps. A request whose metrics are non-finite or negative fails with a nonzero exit code instead of printing a RESULT line.
