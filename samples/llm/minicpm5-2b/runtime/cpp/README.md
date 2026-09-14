[English](README.md) | [简体中文](README_cn.md)

# C++ runtime

Requirements: S600, the supplied OELLM 2.0.4 runtime, C++17 compiler, CMake, gflags, nlohmann-json and curl. On RDK OS use `sudo apt-get install build-essential cmake libgflags-dev nlohmann-json3-dev curl`. Run commands from this directory. `run.sh` verifies/downloads the model, builds the application and sets the library/L2M environment. Set OELLM_RUNTIME_ROOT instead if only the SDK runtime directory was copied to the board.

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

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model_path` | `../../model/s600` | Verified model directory; run.sh resolves its default relative to itself. |
| `--prompt` | `What is 1+1? Give a short answer.` | First user prompt. |
| `--follow_up` | empty | Optional second prompt sharing conversation state. |
| `--max_new_tokens` | 128 | Output limit per request, 1–4096; total context remains 4096. |

Each request prints a `RESULT` JSON line with text, token_ids, status, ttft_ms, decode_tps and e2e_ms. SDK diagnostics may also appear. Status 3 means EOS, 6 means the output limit and 4 means the context limit. Limits are reported as successful bounded requests; invalid input or runtime errors return nonzero. A one-token length-limited request may report zero decode_tps.

`inc/minicpm5.hpp` defines Config, Result and the sequential MiniCPM5 wrapper; `src/minicpm5.cc` implements model setup and generation; `src/main.cc` handles gflags and output. The runtime performs tokenization, BPU execution and decoding. No generated text is executed as code. The temporary runtime JSON is removed after initialization.
