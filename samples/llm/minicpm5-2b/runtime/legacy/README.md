> Board results, accuracy figures and SDK release notes below are records from the source S release; board runs follow the commands in this guide.

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
BOARD=s100 bash ../../model/download_model.sh
BOARD=s100 bash run.sh --build
BOARD=s100 bash run.sh -- --prompt 'What is the capital of France?'
# On S100P:
BOARD=s100p bash ../../model/download_model.sh
BOARD=s100p bash run.sh --build
BOARD=s100p bash run.sh -- --prompt '请用一句话介绍你自己。'
```

Preparation verifies/downloads models; `--build` only builds; ordinary launch only runs the existing binary. `BOARD` is required for this helper. The shared launcher checks the actual target before build or execution. See [launcher options](../README.md).

| Option | Meaning |
|---|---|
| `OELLM_SDK_ROOT` | Extracted S100 1.0.0 SDK root |
| `OELLM_RUNTIME_ROOT` | Runtime directory containing include/lib; overrides SDK_ROOT |
| `MODEL_DIR` | Defaults to `model/$BOARD`; preparation checks existing files against pinned hashes |
| `INFERENCE_TIMEOUT` | Inference timeout in seconds, default 120; excludes download/build |
| `--prompt TEXT` | Defaults to a Chinese self-introduction request |

Success prints the answer and `RESULT status=0 ended=1 failed=0 destroy=0`. Errors return nonzero; timeout returns 124. Use SDK Performance lines only as short-workload measurements. Zero callback performance fields are not measurements.

Manual build and invocation:

```bash
cmake -S . -B build -DMINICPM_TARGET=s100 -DOELLM_RUNTIME_ROOT="$OELLM_SDK_ROOT/oellm_runtime"
cmake --build build --parallel 2
export LD_LIBRARY_PATH="$OELLM_SDK_ROOT/oellm_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
timeout 120 ./build/main --model-path ../../model/s100/minicpm5-2b_ctx4096_s100.hbm   --tokenizer-path ../../model/s100/tokenizer   --template-path ../../model/s100/tokenizer/simple-chat.jinja --prompt 'What is the capital of France?'
```

`inc/minicpm5.hpp` defines the configuration, `prepare_request` (pre-process) and the `RequestOutcome` status record; `src/chat_template.cc` loads and size-checks the chat template outside the inference file; `src/minicpm5.cc` handles SDK initialization, streaming through an injected sink and the single-request teardown; `src/main.cc` parses arguments, injects the stdout sink, prints the RESULT line and maps `RequestOutcome::exit_code`. The inference file performs no console IO of its own. Tokenization, template rendering, BPU execution and sampling use the SDK directly. Each instance serves exactly one request: `init` after `predict` is rejected, so create a new instance or process per request. Streaming tokens reach the sink while the SDK runs; the RESULT line afterwards carries the same status values as before.

Complete native library usage as one self-contained program — copy it, compile against the SDK headers and run:

```cpp
#include <iostream>

#include "minicpm5.hpp"

int main() {
  MiniCPM5Config config;
  config.model_path = "model/s100/minicpm5-2b_ctx4096_s100.hbm";
  config.tokenizer_path = "model/s100/tokenizer";
  config.template_path = "model/s100/tokenizer/simple-chat.jinja";
  config.prompt = "请用一句话介绍你自己。";
  config.text_sink = [](const char* chunk) {  // optional; omit for silent use
    std::cout << chunk << std::flush;
  };
  MiniCPM5 model(config);                    // stores settings only
  model.init();                              // loads model and tokenizer; one request per instance
  RequestOutcome outcome = model.predict();  // streams via the sink
  // consume: outcome.ended, outcome.failed, outcome.sdk_status, outcome.destroy_status,
  // outcome.stream_error; outcome.exit_code() is zero only for normal EOS with cleanup.
  return outcome.exit_code();
}
```

A sink that throws is contained inside the callback: the exception never crosses the vendor C boundary, `stream_error` is recorded on the outcome, streaming stops, and the request still runs to its normal status mapping. END-state and ERROR-state text is suppressed as in the source. `PreparedRequest` owns its prompt and template strings and rebinds its embedded SDK view on copy and move; pass `&request.input` to the SDK only while that instance is alive and unmodified.

## Limits

Fixed chunk=256 and cache=4096; input and output share the context budget. This entry point uses a process timeout because it has no usable output-token limit in the old API. The separate [full evaluator](../../evaluator/legacy/README.md) covers PPL, two-turn conversation, long input and 50 repeated requests. Current PPL and reference-text matching do not meet the ≤3% target; see [results](../../evaluator/README.md). This CLI is single-request; tool calls, multimodal input and long-duration stability are outside its scope.

Legacy BPE merges use strings and a simplified non-thinking template. The deployment primary EOS is the existing `<|im_end|>` (130073). Preparation preserves the original checkpoint. A single request uses request_id=0 as in the SDK demo.
