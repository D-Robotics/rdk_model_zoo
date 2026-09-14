[English](README.md) | [简体中文](README_cn.md)

# S100 / S100P: OELLM 1.0.0 C++ runtime

This entry point streams one greedy, non-thinking text response. `runtime/cpp` uses the separate S600 OELLM 2.0 API. Do not mix their SDKs or artifacts.

## Dependencies and memory

Install `build-essential cmake curl` on the board. Obtain [S100 SDK 1.0.0](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s100/1.0.0/D-Robotics_LLM_S100_1.0.0_SDK.tar.gz) separately: this sample needs `oellm_runtime/include/xlm.h` and `oellm_runtime/lib/`. Tested UCP/DNN 3.7.3 and HBRT 4.2.11. Libraries are selected with `LD_LIBRARY_PATH`, without overwriting system libraries. No board-side PyTorch/compiler Python installation is required.

The approximately 2.9 GB HBM requires sufficiently large contiguous BPU memory. Both tested boards used the following `/boot/config.txt` settings, verified after reboot:

```ini
ion=ion_cma_size=0x40000000
ion=ion_reserved_size=0xf0000000
ion=ion_carveout_size=0xf0000000
```

This allocates CMA 1 GiB and reserved/carveout 3.75 GiB each. Back up the original configuration and check your board capacity before changing it. **run.sh never changes boot settings or reboots.** Linux reported approximately 2.8 GiB on the tested S100 and 14 GiB on S100P; these are not universal hardware capacities. Close unnecessary S100 applications. `free` does not establish contiguous BPU heap capacity. Allow about 6 GB free storage for download and extraction.

## Run

```bash
export OELLM_SDK_ROOT=/path/to/D-Robotics_LLM_S100_1.0.0_SDK
cd samples/llm/minicpm5-2b/runtime/legacy
BOARD=s100 bash run.sh --prompt 'What is the capital of France?'
# On S100P:
BOARD=s100p bash run.sh --prompt '请用一句话介绍你自己。'
```

The script verifies the model, builds with CMake and runs inference. `BOARD` defaults to `s100`; explicitly use `s100p` on S100P. There is no automatic board detection or cross-board artifact compatibility.

| Option | Meaning |
|---|---|
| `OELLM_SDK_ROOT` | Extracted S100 1.0.0 SDK root |
| `OELLM_RUNTIME_ROOT` | Runtime directory containing include/lib; overrides SDK_ROOT |
| `MODEL_DIR` | Defaults to `model/$BOARD`; existing files must pass pinned hashes |
| `INFERENCE_TIMEOUT` | Inference timeout in seconds, default 120; excludes download/build |
| `--prompt TEXT` | Defaults to a Chinese self-introduction request |

Success prints the answer and `RESULT status=0 ended=1 failed=0 destroy=0`. Errors return nonzero; timeout returns 124. Use SDK Performance lines only as short-workload measurements. Zero callback performance fields are not measurements.

Manual build and invocation:

```bash
cmake -S . -B build -DOELLM_RUNTIME_ROOT="$OELLM_SDK_ROOT/oellm_runtime"
cmake --build build --parallel 2
export LD_LIBRARY_PATH="$OELLM_SDK_ROOT/oellm_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
timeout 120 ./build/main --model-path ../../model/s100/minicpm5-2b_ctx4096_s100.hbm   --tokenizer-path ../../model/s100/tokenizer   --template-path ../../model/s100/tokenizer/simple-chat.jinja --prompt 'What is the capital of France?'
```

`inc/minicpm5.hpp` defines configuration and model ownership; `src/minicpm5.cc` handles the SDK lifecycle/callback; `src/main.cc` parses arguments. Tokenization, template rendering, BPU execution and sampling use the SDK directly.

## Limits

Fixed chunk=256 and cache=4096; input and output share the context budget. This entry point uses a process timeout because it has no usable output-token limit in the old API. Multi-turn, tools, multimodal input, full PPL and long-duration stability are unverified.

Legacy BPE merges use strings and a simplified non-thinking template. The deployment primary EOS is the existing `<|im_end|>` (130073). Preparation preserves the original checkpoint. A single request uses request_id=0 as in the SDK demo.
