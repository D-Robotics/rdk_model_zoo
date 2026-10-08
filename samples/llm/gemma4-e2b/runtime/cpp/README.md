# C++ Runtime

[中文](./README_cn.md) | **English**

C++ inference runtime for Gemma4-E2B VLM on D-Robotics RDK S100P and S600. It loads pre-compiled HBM models built for the matching SoC and runs real-time Vision-Language inference on the BPU.

> Part of [Gemma4-E2B sample](../../README.md). Full upstream project: [gemma4-e2b-rdk-s100p](https://github.com/shockley6668/gemma4-e2b-rdk-s100p).

<a id="overview"></a>
## C++ inference

Run multi-turn text or image-and-text chat with Gemma4-E2B on S100P/S600. The native application loads Vision before Text, keeps both models resident and streams generated text.

<a id="directory"></a>
## Directory structure

```text
cpp/
├── inc/  # Public C++ interfaces
├── src/  # C++ runtime and command-line entry
├── CMakeLists.txt  # Native build configuration
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── build.sh  # Shell command
├── launcher.py  # Python script
└── run.sh  # Run the sample
```

<a id="supported-boards"></a>
## Boards and runtime scope

S100P uses `nash-m` HBMs and S600 uses `nash-p` HBMs. The S100 `nash-e` runtime branch requires separately supplied matching HBMs.
`--target` only selects a target; it does not convert an existing model.
Hosts can run launcher help and preview; native executables require the board SDK.

<a id="dependencies"></a>
## Prerequisites

The board must have the OE-LLM runtime installed:

```bash
# Verify Horizon BPU SDK
ls /usr/hobot/lib/libdnn.so    # BPU inference lib
ls /usr/hobot/lib/libhbucp.so  # Memory management lib
ls /usr/include/hobot/dnn/hb_dnn.h
```

System dependencies (usually pre-installed on OE-LLM images):

```bash
sudo apt install cmake g++ libopencv-dev libgflags-dev nlohmann-json3-dev cargo wget git curl
```

> The launcher requires Python 3 standard library only. Inference and tokenization remain native C++ via explicitly prepared `tokenizers-cpp`. Direct native binary execution does not use Python.

## Directory Layout

```
runtime/cpp/                            C++ source code (this directory)
├── CMakeLists.txt                      Build entry (pulls in tokenizers-cpp + gflags)
├── run.sh                              Explicit build or native launch
├── inc/                                Public headers
│   ├── gemma4_config.hpp               Model constants (image token IDs, dims, ...)
│   ├── gemma4_chat_app.hpp             Interactive chat application session (facade)
│   ├── gemma4_text_engine.hpp          Text orchestrator (prefill + decode + KV session)
│   ├── gemma4_text_inputs.hpp          Text stage 1: prepared CPU inputs (ids/embeds/positions/masks)
│   ├── gemma4_text_transport.hpp       Text stage 2: raw SDK writes/inference/KV collection
│   ├── gemma4_text_session.hpp         Text session state and continuation policy
│   ├── gemma4_text_tensor.hpp          Fixed Text export descriptor contract and strided IO
│   ├── gemma4_vision_engine.hpp        Vision ViT engine
│   ├── gemma4_embeddings.hpp           Token embedding lookup + vision injection
│   ├── gemma4_kv_cache.hpp             Zero-copy KV cache management
│   ├── gemma4_vision_preprocess.hpp    Image resize + patchify
│   ├── gemma4_vision_task.hpp          Vision stages and explicit runner composition
│   ├── gemma4_image_io.hpp             Application image IO
│   ├── gemma4_vision_tensor.hpp        SDK descriptors and strided packing/extraction
│   ├── gemma4_vision_debug.hpp         Optional diagnostics
│   ├── gemma4_native_tokenizer.hpp     Native C++ tokenizer (from OE-LLM-s600)
│   ├── gemma4_tokenizer.hpp            TokenizerBridge: chat template + image expand
│   └── hb_utils.hpp                    Horizon BPU helpers (tensor, flush, infer)
└── src/                                Implementation + executables
    ├── main.cpp                        ★ Thin entry: flags → paths → app construct & run
    ├── gemma4_chat_app.cpp             ★ Interactive chat session (REPL, history, console IO)
    ├── gemma4_server.cpp               HTTP API server
    ├── gemma4_demo.cpp                 Single-shot VLM demo
    ├── gemma4_text_bench.cpp           Text-only benchmark
    ├── gemma4_golden_verify.cpp        Golden mask/KV alignment checker
    └── gemma4_*.cpp                    Engine implementations

../../third_party/
└── tokenizers-cpp/                     Explicitly prepared (see third_party/README.md)
```

<a id="build"></a>
## Build

Prepare and build explicitly from the repository root (install system packages above first):

```bash
cd samples/llm/gemma4-e2b
export GEMMA4_HOME=~/gemma4_e2b_s600
# S100P: s100p; S600: s600. Keep different targets in separate model directories.
GEMMA4_SOC=s600 bash model/download_model.sh
bash third_party/install_tokenizers_cpp.sh
cd runtime/cpp
./run.sh --target s600 --build
./run.sh --target s600
./run.sh --target s600 server --port=8000
./run.sh --target s600 main --max_tokens=512
```

Place launcher options before the app name and native flags after it, e.g. `./run.sh --target s600 demo text --prompt "Hello"`.
Zero arguments still select `main`, which must already be built; `--build` builds only and never starts inference.
`--home` defaults to `GEMMA4_HOME` or `~/gemma4_e2b`; `--build-dir` defaults to this directory's `build/`.
The default `--target auto` resolves actual hardware through the shared platform registry, including S100P aliases.
An explicit target mismatch rejects execution. S100 retains the manual-HBM route without a default public HBM.
Use separate model directories for different targets, whose HBM filenames are identical.

Offline host entry points:
```bash
./run.sh --help
./run.sh --target s600 --dry-run demo text --prompt "Hello"
./run.sh --target s100p --build --dry-run
```
Preview prints the selected target, planned command and settings as JSON with `executed=false`.
Real commands propagate the native exit code; launcher preflight errors return 2 with a missing-binary or target message.

Dependency preparation needs Git/network access and an explicitly installed stable Rust 1.80+ toolchain; it never installs or upgrades Rust.
Neither the launcher nor CMake calls that installer automatically. Compiling Rust dependencies may still access package registries;
offline builds require pre-populated dependency caches.
Use `GEMMA4_ABSL_PREFIX=/opt/abseil ./run.sh --target s600 --build` for a separately installed Abseil package.

This produces 5 executables in `build/`:

| Binary | Description |
|--------|-------------|
| `main` | Interactive VLM chat with streaming output (primary entry) |
| `gemma4_server` | HTTP API server for programmatic access |
| `gemma4_demo` | Single-shot: image + prompt → text |
| `gemma4_text_bench` | Text-only inference benchmark |
| `gemma4_golden_verify` | Verify prefill tensors against golden data |

## Download Pre-compiled Models

```bash
export GEMMA4_HOME=~/gemma4_e2b_s600
GEMMA4_SOC=s600 bash ../../model/download_model.sh
```

On S100P and S600 this downloads the validated public HBM files plus the
shared embedding and tokenizer assets. On S100, pre-place matching HBMs or
set `GEMMA4_MODEL_BASE_URL`; missing shared assets are still downloaded.

```
~/gemma4_e2b/
├── model/
│   ├── gemma4-e2b_vit_ptq.hbm                          # 329-377 MB Vision
│   ├── gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm      # 4.5 GB  Text
│   └── tok_embeddings.bin                               # 1.5 GB  Embedding
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

<a id="run"></a>
## Run

After building, enter `samples/llm/gemma4-e2b/runtime/cpp/build` from the repository root. The direct native commands below assume this working directory. Set `GEMMA4_HOME` to the matching target models:

```bash
export GEMMA4_HOME=~/gemma4_e2b_s600

# Manual S600 launch: run.sh applies these settings automatically
unset LD_LIBRARY_PATH GEMMA4_USE_DNN_V3
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6

# Interactive VLM chat (zero-arg default uses $GEMMA4_HOME)
./main

# Inside the chat:
#   /image /path/to/photo.jpg        Load an image
#   What do you see in this image?   Ask a question
#   /context                          Show KV-cache usage
#   /reset                            Clear conversation
#   /quit                             Exit
```

Example output:

```
gemma4> /image test.jpg
Processing image: test.jpg...
Image loaded (430080 features).
gemma4> Describe this image
This is a photograph of a Red Panda resting on a wooden structure...
```

### 4096-token context

- The Text HBM has a fixed 4096-token capacity, so `prompt_tokens + output_tokens <= 4096`.
- `--max_tokens=0` is the default for both `main` and `gemma4_server`. It gives each turn all capacity remaining after the prompt, so a short prompt can receive an output budget close to 4096 tokens.
- The next turn reserves at least `--min_response_tokens` tokens (256 by default). When needed, the oldest complete user/assistant pairs are removed and the KV cache is rebuilt.
- `/context` reports used tokens, remaining capacity, and turn count. Stop tokens are neither printed nor stored in assistant text.
- `main` loads both Text and Vision before entering the interactive loop and keeps both resident for the process lifetime. S100, S100P, and S600 share this lifecycle; `/image` only runs image preprocessing and Vision inference and never reloads either model.
- Multimodal follow-ups retain the original image turn and explicitly inject the same Vision features beside the latest user question, with at most two 280-token image blocks in the prompt.
- Internal diagnostics are quiet by default. `[VLM-FIX]` output follows `GEMMA4_DEBUG=1`; Text engine diagnostics require an installed `SetDebugSink` receiver.

<a id="parameters"></a>
## Command-line Parameters

All five binaries use [gflags](https://github.com/gflags/gflags) for argument
parsing. Flag names use `snake_case` per the Model Zoo guideline. Every flag
has a default that makes the binary runnable with zero arguments once
`GEMMA4_HOME` is exported.

### `main` — interactive VLM chat

| Flag | Type | Default | Description |
|---|---|---|---|
| `--text_hbm` | string | `$GEMMA4_HOME/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm` | Path to text LLM HBM |
| `--vision_hbm` | string | `$GEMMA4_HOME/model/gemma4-e2b_vit_ptq.hbm` | Path to vision ViT HBM |
| `--tok_embeddings` | string | `$GEMMA4_HOME/model/tok_embeddings.bin` | External token embedding table |
| `--tokenizer_path` | string | `$GEMMA4_HOME/tokenizer/tokenizer.json` | HF tokenizer JSON |
| `--max_tokens` | int | `0` | Max new tokens per turn; `0` uses all KV capacity remaining after the prompt |
| `--min_response_tokens` | int | `256` | Minimum reply capacity preserved while trimming old history |

### `gemma4_demo` — single-shot text or VLM

```
./gemma4_demo {text|vlm} --prompt "..." [--image_path PATH] [other flags]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--text_hbm` | string | same as `main` | Text LLM HBM |
| `--vision_hbm` | string | same as `main` | Vision ViT HBM (vlm only) |
| `--tok_embeddings` | string | same as `main` | Token embedding table |
| `--prompt` | string | `""` (required) | User prompt text |
| `--image_path` | string | `""` | Image path (required when mode = `vlm`) |
| `--max_tokens` | int | `32` | Max new tokens |

### `gemma4_server` — OpenAI-compatible text chat

`gemma4_server` keeps the Text HBM resident and exposes a serialized OpenAI-compatible HTTP API. Consecutive requests reuse the KV cache when their token prefixes match. The endpoint is text-only; image messages return HTTP 400 and should be handled with the interactive `main` executable.

~~~bash
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh server --host=0.0.0.0 --port=8000
~~~

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Readiness, model id, context size, and cached-token count |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `POST` | `/v1/chat/completions` | Non-streaming JSON or SSE streaming chat completion |

Non-streaming request:

~~~bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-e2b","messages":[{"role":"user","content":"Write a long introduction to the RDK S600."}],"max_tokens":0}'
~~~

`max_tokens: 0` is a sample extension meaning “use every token remaining in the fixed 4096-token KV cache.” For SSE output, add `"stream": true`; `stream_options.include_usage` is also supported.

For ChatBox, set the API type to OpenAI-compatible, the base URL to `http://BOARD_IP:8000/v1`, and the model to `gemma4-e2b`. If the client requires an API key, any non-empty placeholder is accepted because the server does not validate the `Authorization` header.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--host` | string | `0.0.0.0` | HTTP listen address |
| `--port` | int | `8000` | HTTP listen port |
| `--model` | string | `gemma4-e2b` | Model id exposed by `/v1/models` |
| `--text_hbm` | string | same as `main` | Text LLM HBM |
| `--tok_embeddings` | string | same as `main` | Token embedding table |
| `--tokenizer_path` | string | same as `main` | HF tokenizer JSON |
| `--max_tokens` | int | `0` | Default output limit; `0` uses all capacity remaining after the prompt |
| `--min_response_tokens` | int | `256` | Reply capacity preserved while trimming old complete turns |
| `--request_limit_mb` | int | `4` | Maximum HTTP request body size |

### `gemma4_text_bench` — text-only throughput / smoke test

```
./gemma4_text_bench {bench|generate} [flags]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--text_hbm` | string | same as `main` | Text LLM HBM |
| `--tok_embeddings` | string | same as `main` | Token embedding table |
| `--token_ids` | string | `9259` (= `Hello`) | Prompt token ids, comma-separated |
| `--max_tokens` | int | `8` | New tokens to generate |
| `--warmup` | int | `2` | Decode warmup steps before timing |

### `gemma4_golden_verify` — prefill golden alignment check

| Flag | Type | Default | Description |
|---|---|---|---|
| `--golden_root` | string | `$GEMMA4_HOME/golden_mask_kv` | Root dir of golden tensors |
| `--prompt_id` | string | `prompt_0` | Prompt sub-directory |
| `--text_hbm` | string | same as `main` | Text LLM HBM |
| `--tok_embeddings` | string | same as `main` | Token embedding table |

Pass `--help` to any binary to see the gflags-generated full help.

<a id="interface-lifecycle"></a>
## Key Design Decisions

1. **Vision injection is raw** — ViT output `[280, 1536]` is injected directly into `inputs_embeds` at image soft-token positions (token ID 249560). No L2-norm scaling, no √1536 multiplication.

2. **PLE uses pad embedding** — At image positions, the Per-Layer Embedding token-identity path uses `pad_token_id=0` (not 249560), matching HuggingFace's `masked_scatter` behavior.

3. **Chat template** — Prompts are formatted in C++ to the Gemma turn format (`<bos><|turn>user\n...<turn|>\n<|turn>model\n`), matching `chat_template.jinja`. Tokenization uses the native `tokenizers-cpp` (HF tokenizers), not Python.

4. **Zero-copy KV cache** — KV cache memory is allocated once and shared between prefill and decode via pointer assignment, avoiding per-step memcpy.

5. **Chunked prefill** — Prompts longer than `chunk_size=256` tokens are automatically split into multiple prefill chunks.

6. **Full KV budgeting** — The interactive entry computes the output limit from the current prompt, can use the cache through `4096/4096`, and trims history by complete turns on the next request.

7. **Unified dual-model lifecycle** — `main` always loads Vision before Text and keeps both resident for the process lifetime. This order avoids the S600 cross-core IOVA mapping conflict while preserving identical chat control flow on S100, S100P, and S600; only the matching HBMs, CMake SoC macros, and `run.sh` environment setup differ.

### Interactive chat entry: application facade vs. model classes

`main` is split at the application boundary. `main.cpp` is the thin entry: it parses
gflags, resolves `$GEMMA4_HOME` defaults, validates the generation flags, then
constructs `gemma4::chat::InteractiveChatApp` (`gemma4_chat_app.hpp/.cpp`) and calls
`Run`. The app facade owns everything console- and session-shaped — banner/help,
REPL prompts, UTF-8/GB18030 terminal normalization, chat-history JSON, oldest-turn
trimming against the 4096-token budget, prefix-mismatch resets and the streaming
echo. It performs no model math: text generation is delegated to the real runtime
classes `TextEngine::ContinueGenerateStream` (with `BuildPromptHidden` for image
turns) and image encoding to `PredictVision` + `VisionEngine::Infer`. The facade
is an application class, not a model class — no `predict`-style engine API
performs console IO, and the engines themselves never print implicitly.

### Vision library interfaces and responsibilities

`gemma4_image_io` reads files; `gemma4_vision_preprocess` transforms in-memory pixels;
`gemma4_vision_task` composes the stages; `VisionEngine` owns SDK model/tensor transport.
Interactive and single-shot VLM entry points use the same composition.

| Interface | Input → output | Contract |
| --- | --- | --- |
| `LoadImage(path)` | File path → `cv::Mat` | Application IO, OpenCV BGR decoding; throws on read failure |
| `PreprocessImage(bgr)` | Nonempty 2D `CV_8UC3` → float `[2520,768]` | BGR→RGB, bicubic resize to 960×672, divide by 255, 16×16 patches; no input mutation or file IO |
| `ForwardVision(patches, runner)` | Prepared float patches → raw runner result | Fixed element count, finite `[0,1]` values, exactly one explicit runner call |
| `PostprocessVision(raw)` | Float `[280,1536]` → owned feature vector | Count and finite-value checks; no scaling, L2 or extra normalization |
| `PredictVision(bgr, runner)` | BGR image and runner → features | Three-stage composition; no SDK construction, file IO or printing |

Patch rows precede patch columns; each patch contains pixel rows, pixel columns and interleaved RGB channels.
Results own their values, so later calls cannot overwrite retained outputs. Serialize access to the SDK engine.
`VisionEngine::Infer` takes prepared patches (prepare image paths through the explicit IO below).

This complete example links `gemma4_runtime` in an SDK-enabled project:

```cpp
#include <iostream>
#include <stdexcept>
#include <vector>
#include "gemma4_image_io.hpp"
#include "gemma4_vision_engine.hpp"
#include "gemma4_vision_task.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: vision_example VISION_HBM IMAGE\n";
    return 2;
  }
  try {
    const cv::Mat image = gemma4::LoadImage(argv[2]);
    gemma4::VisionEngine engine(argv[1]);
    const auto features = gemma4::PredictVision(
        image, [&engine](const std::vector<float>& patches) {
          return engine.Infer(patches);
        });
    std::cout << features.size() << " vision feature values\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
```

The model must match the board and build target. Success prints `430080 vision feature values`; this example reports Vision features rather than generating text.
Save it as `vision_example.cpp` in the native CMake project and add:

```cmake
add_executable(vision_example vision_example.cpp)
target_link_libraries(vision_example PRIVATE gemma4_runtime)
```

<a id="results-interpretation"></a>
## Verification

To verify board inference matches the PC golden data:

```bash
# Optional verification data: place golden_mask_kv/ under
# $GEMMA4_HOME/golden_mask_kv/. It is not included in the public model archive.
./gemma4_golden_verify --prompt_id prompt_0
# Expected: ALL PASSED (five input comparisons satisfy their individual criteria)
```

`main` and `demo` print generated text; `server` returns JSON or SSE; `text_bench` prints generation/throughput records.
For dataset-level accuracy, use the evaluator's PC BC comparison. The golden verifier compares five prefill inputs: exact integer equality,
embedding maximum absolute error ≤1e-3, and zero error for both masks. Cosine is printed for reference only.
`ALL PASSED` corresponds to exit code 0; mismatches or exceptions return 1. See [evaluation prerequisites](../../evaluator/README.md).
Each TextEngine owns one session and its KV state; callers must serialize access. An interactive session is not a stateless, concurrently shared inference function.

### Host regression without the SDK

Checks stage contracts and source preprocessing parity using already installed OpenCV C++ libraries; it installs no dependencies and prepares no models:

```bash
# From repository root; set OpenCV_DIR if OpenCV is not on CMake's search path.
cmake -S samples/llm/gemma4-e2b/tests/native -B /tmp/gemma-vision-tests -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/gemma-vision-tests --parallel
ctest --test-dir /tmp/gemma-vision-tests --output-on-failure
```

The suite covers Vision stage/source-image/tensor checks, KV
allocation/reset/append/aliasing/prefix-retention, Text ownership (including
tensor adoption under injected allocation failure), tensor contracts,
generation flow and stage/session behavior, plus the runnable README example.
Assertions stay enabled in Release builds. The tests run against an explicit
offline runner; real SDK descriptors and board numerical results come from
the board commands in this guide.

The interactive chat entry has its own host check, which compiles the production
`src/gemma4_chat_app.cpp` and the real `src/main.cpp` against the engine doubles in
`tests/native/chat_app_doubles.cpp`, the SDK header double in `tests/native/sdk_fixtures`
and clearly-marked compile stubs for the third-party tokenizers-cpp / OpenCV headers
(`tests/native/app_stubs/`), then drives the REPL through redirected stdin:

```bash
python3 -m unittest discover -s samples/llm/gemma4-e2b/tests -p test_cpp_chat_app.py -v
```

The check covers the session logic end to end: engine construction messages,
streaming echo, `/reset` `/context` commands, image-turn wiring
(`LoadImage`/`PredictVision`/`Infer` chain with prompt-hidden injection),
oversize-prompt rejection, oldest-turn history trimming, per-turn rebuild
mode, cross-turn context growth, GB18030 terminal conversion, and the thin
entry's flag validation. It runs against marked doubles; the real
SDK/OpenCV/tokenizers-cpp stack and generation run on the board.
`GEMMA_CXX` selects the compiler. nlohmann-json headers come from the
`GEMMA_JSON_INCLUDE` override, `pkg-config nlohmann_json`, or standard system
include roots (`/usr/include`, `/usr/local/include`, `/opt/homebrew/include`);
the entry check likewise links the real host gflags from
`GFLAGS_INCLUDE_DIR` + `GFLAGS_LIB_DIR` (both together), `pkg-config gflags`,
or standard system roots. A missing dependency skips the affected host checks
with the recorded reason, while an invalid override fails the run. iconv links `-liconv` only on macOS; Linux
uses the libc iconv.

### SDK failure handling

Failed Vision construction releases acquired input/output buffers and the packed model; successful SDK calls returning null handles/buffers are rejected.
`MakeTensor` also releases memory returned alongside an allocation error. Vision requires exactly one input and one output; tensor type/shape/stride checks are described below.

Full-flush and Text/KV selective-flush entries share one task lifecycle: input flush → infer → compiled-core scheduling → submit/wait → output flush/property refresh → release.
Failures after task acquisition release it, including inference errors that still return a task. Normal-path release errors propagate without retrying the same handle.
Selective-flush index semantics, S600 compiled-core selection and optional V3 dispatch follow the source implementation.

Host resource tests run against SDK doubles on both the S100/S600 compile
branches and check memory/handle ownership under error injection. From the
repository root:

```bash
python3 -m unittest discover -s samples/llm/gemma4-e2b/tests -p test_cpp_resources.py -v
```

### Vision tensor transport contract

`gemma4_vision_tensor` owns physical descriptor validation, F16 storage packing and strided output extraction; `VisionEngine` composes these operations with SDK calls.
Descriptors are validated before allocation and output properties are revalidated after inference. A refreshed allocation length does not replace the original buffer capacity.

| Item | Accepted contract |
| --- | --- |
| Input | `F16`, no quantization metadata, logical matrix `[2520,768]` |
| Output | `F16` or `F32`, no quantization metadata, logical matrix `[280,1536]` |
| Shape | Optional leading singleton axes such as `[1,2520,768]`; no extra batch, transpose or unrelated equal-element-count shapes |
| Stride | Bytes aligned to element width, nonoverlapping elements/rows, every accessed address within declared allocation and original buffer capacity |
| Data | Prepared float patches must be finite in `[0,1]`; output NaN/Inf is rejected |

Input applies the source F32→F16 truncation and zeroes padding before writing. Output extraction honors both row and column strides and returns owned floats without padding.
Output tensors must use a supported F16/F32 dtype; unknown and integer outputs are rejected. F16/F32 storage conversion does not modify the quantization recipe.

Host integration tests call the production `VisionEngine::Infer`; SDK doubles
inspect packed inputs and populate F16/F32 outputs with row and element gaps,
and inject invalid type, quantization flags, shape, stride, post-inference
capacity changes and nonfinite output. Validate real published HBM
descriptors through the board run.

### KV state and ownership

One `KvCache` owns 30 UCP buffers for 15 K/V layers in one serialized session. Each layer is a contiguous S8 `[4096, head_dim]`
matrix, with head dimensions fixed to 256 or 512 by `kHeadDims`. Trailing allocation padding is supported; internal row padding is not.
Allocation size is not a token-row count, so trailing padding does not participate in rolling or prefix retention.

| Operation | State and aliases |
| --- | --- |
| `Allocate(k_bytes, v_bytes)` | Exactly 15 sizes each, at least the logical matrix size; replaces buffers/resets positions only after complete success; failure preserves old buffers/state |
| `Reset` | Zeroes K/V using their own capacities and clears positions without freeing or changing addresses; bound input aliases remain valid |
| `AppendPrefillChunk(...)` | Appends 1–256 contiguous token positions starting at `OccupiedLen`; validates all layer pointers, strides and positions before cache mutation |
| `AppendDecodeStep(...)` | Same append path for one row |
| `CompactShift(n_keep, discard)` | Retains the first `n_keep` resident rows, discards the suffix and renumbers positions from zero; `discard` must equal old logical length minus `n_keep` |
| `PhysicalIndex(pos)` | Right-aligned physical row, or -1 if not resident; `OccupiedLen` is the logical end and `CacheStart` the oldest resident position |

Successful reallocation invalidates old aliases and must happen before binding model inputs; failed allocation preserves them.
Append source output memory must be separate from the cache and readable for at least `(rows-1)*row_stride + head_dim` bytes;
append rejects source pointers inside any resident K/V allocation directly. It also requires equal K/V output row strides
per layer, which the Text descriptor layer enforces before every append. Shared input buffers do not mean zero CPU movement:
append and prefix retention move/copy rows, and the inference entry flushes CPU-modified KV inputs.

Maintain cache data and positions with `Reset`, `Append` and `CompactShift`.
`TextEngine::ContextShift` retains a prefix; callers then re-prefill the suffix they want.

Independent host test entry:
```bash
python3 -m unittest discover -s samples/llm/gemma4-e2b/tests -p test_cpp_kv.py -v
```

### Text initialization and tensor ownership

`ModelIo` is a move-only owner of one subgraph's input/output allocations. Its
subgraph handle is borrowed from the packed HBM; KV input slots explicitly borrow
memory from `KvCache`. Moving an owner transfers those flags with the tensors.
Partial construction releases completed allocations. TextEngine clears both
subgraphs before releasing the packed model, including when initialization fails.
A constructor failure propagates to the application; no partially initialized
engine is returned. The embedding loader runs before model acquisition.

The fixed text export requires 35 inputs (five ordinary inputs and 15 K/V pairs)
and 31 outputs (logits and 15 K/V pairs). Null model handles, incompatible counts
and a missing/nonpositive sequence dimension are rejected before indexed use.

### Text tensor transport contract

`gemma4_text_tensor` owns physical descriptor validation, strided input packing,
KV output row addressing and greedy logits argmax; `TextEngine` composes these
operations with SDK calls. Every descriptor is validated against the fixed export
before any allocation; output descriptors are revalidated after each inference
against the capacity the engine originally allocated, not the refreshed claim.
A mismatched export, such as one with 512 positions or an unsupported dtype,
is rejected at construction before any tensor is read.

| Binding | Accepted contract |
| --- | --- |
| `inputs_embeds` | `F32`, no quantization metadata, matrix `[chunk,1536]` (prefill) / `[1,1536]` (decode) |
| `token_ids` / `position_ids` | `S64` / `S32`, one row of `seq` elements (source graph declares `[1,seq]`) |
| `full_mask` / `sliding_mask` | `S16`, matrix `[seq,4096]`; row padding allowed, element gaps rejected |
| `logits` | `S16`, matrix `[seq,262144]`; row and column strides honored, other widths never reinterpreted |
| K/V inputs (5..34) | `S8`, dense `[4096,head_dim]` per layer (singleton axes ignored); internal row padding rejected |
| K/V outputs (1..30) | `S8`, `[seq,head_dim]` per layer; row padding allowed, K/V row strides must agree |

Singleton axes are collapsed before comparison, so `[4096,1,head_dim]`,
`[1,seq]` and `[seq,1536]` declarations of the same physical layout are all
accepted. Byte strides must be element aligned, nonoverlapping, and every
addressed byte must stay inside the declared allocation and the original
buffer capacity. Sequence dimensions are pinned: prefill 256, decode 1 — the
`kChunkSize`/`kCacheLen`/`kHiddenSize`/`kVocabSize`/`kHeadDims` constants are
the export contract, and a differently exported model needs an explicit
adapter instead of silently entering this engine.

Mask int16 quantization, logits `kLogitScale` dequantization and int8 KV
storage stay CPU-side source algorithms: descriptors carrying quantization
metadata are rejected, and the write path zeroes padding before copying
through the descriptor strides. Greedy argmax scales int16 storage by
`kLogitScale` and keeps the first maximum; ties and all-zero rows therefore
behave exactly as the source implementation.

Host coverage: a helper contract test (padding, singleton axes, dtype/shape/
stride/capacity/quantization rejections, argmax row addressing and tie
semantics), and a generation-flow test that drives the production engine
against an SDK double speaking this contract — generated tokens, cache
transport through the borrowed KV inputs, inference failure cleanup,
post-inference descriptor drift and a constructor rejection matrix.

Host test entry:
```bash
python3 -m unittest discover -s samples/llm/gemma4-e2b/tests -p test_cpp_text_tensors.py -v
```

Host tests replace only SDK calls and embedding loading: injected failure
points, six invalid descriptor cases and normal teardown check that no tensor/model allocation remains. A
separate owner test covers partial construction, moves, repeated clearing and
borrowed-cache survival. No weights are loaded and no inference calls are made
in the ownership tests.

### Text pipeline stages

The Text pipeline is separated into three explicit stages plus a session
policy module; `TextEngine` only sequences them. The behavior follows the source
implementation — greedy `kLogitScale` decoding, first-max ties, the
EOS/turn-end set, full return vectors, prefix-continuation alignment and
benchmark timing scope — with each responsibility in one addressable unit:

| Stage | Header | Responsibility |
| --- | --- | --- |
| 1. Input preparation | `gemma4_text_inputs.hpp` | `PrepareBatchInputs` / `PrepareDecodeInputs` build a `TextBatchInputs` value (PLE-substituted ids, embedding rows, positions, quantized masks) with plain vectors — no SDK types. Also owns the source mask builders. |
| 2. SDK transport | `gemma4_text_transport.hpp` | `InitTextSubgraph` (descriptor contract + allocation), `BindKvCache` (zero-copy borrowing), `WriteBatchInputs` (strided writes), `RunSubgraphInference` (flush → infer → refresh), `CollectKvOutputs` (revalidated KV rows). No decoding, no IO. |
| 3. Decode + KV update | `gemma4_text_engine.cpp` | Explicit steps: `ArgmaxTextLogits` decodes, `KvCache::Append*` updates the cache, then session counters advance. |
| Session policy | `gemma4_text_session.hpp` | Pure decisions over `TextSessionState`: context shift, auto-truncate, continuation alignment, logits row selection. |

Extent and window contracts (validated with overflow-safe signed checks
before any allocation, lookup or write; violations throw
`std::invalid_argument`/`std::runtime_error` instead of clamping attention):

- Mask geometry (stage 1 and the public mask helpers): `0 <= chunk_valid <=
  seq_len`, `1 <= seq_len <= 4096` and `chunk_start + seq_len <= 4096`. A
  chunk or decode position beyond the fixed 4096-token context is an error —
  `AutoTruncate`/`ContextShift` are the tools to stay inside the window.
- Prepared token count: exactly `chunk_valid` ids per chunk.
- Prebuilt hidden: the continuation entry points and `GenerateWithPrompt
  Embeddings` require exactly `full_ids.size * kHiddenSize` floats indexed
  from the prompt start (not a suffix); the rejection happens before any
  session state changes. `PrepareBatchInputs` itself refuses a hidden smaller
  than the rows it indexes.
- Continuation entry points validate the hidden extent before the alignment
  shift, so a rejected call leaves the session reusable.

The engine prints nothing on its own. `SetDebugSink` installs an explicit
receiver for Text engine diagnostics. `[VLM-FIX]` diagnostics follow
`GEMMA4_DEBUG=1`.

The example takes the Text HBM path and embedding path as its two arguments.
Prepare `model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm` and
`model/tok_embeddings.bin` under `GEMMA4_HOME` first. Include the runtime's
`inc` directory and link its Text engine sources with the matching DNN/UCP SDK,
as in the CMake build above. The token IDs below illustrate the tensor API;
for chat text, obtain IDs with `TokenizerBridge::EncodeMessagesJson`.

```cpp
#include "gemma4_text_engine.hpp"
#include "gemma4_text_inputs.hpp"
#include <stdexcept>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  // Stage setup: subgraph owners plus one borrowed KV cache.
  hbDNNPackedHandle_t packed = nullptr;
  const char *model_file = argv[1];
  if (hbDNNInitializeFromFiles(&packed, &model_file, 1) != 0) return 2;
  gemma4::TokenEmbeddings embeddings(argv[2]);
  gemma4::ModelIo prefill = gemma4::InitTextSubgraph(packed, "prefill", gemma4::kChunkSize);
  gemma4::ModelIo decode  = gemma4::InitTextSubgraph(packed, "decode", 1);
  gemma4::KvCache cache;
  gemma4::BindKvCache(prefill, decode, cache);  // KV slots borrow the cache

  const std::vector<int64_t> prompt = {11, 22, 33, 44, 55};
  // Stage 1: prepared per-call context.
  const auto batch = gemma4::PrepareBatchInputs(
      embeddings, prompt, 0, static_cast<int>(prompt.size()), nullptr,
      gemma4::kChunkSize);
  // Stage 2: strided write, then one selective-flush inference.
  gemma4::WriteBatchInputs(prefill, batch);
  gemma4::RunSubgraphInference(prefill);
  // Stage 3: append validated KV rows, then decode greedily.
  const auto rows = gemma4::CollectKvOutputs(prefill, 5);
  // (append rows via cache.AppendPrefillChunk; see the runnable example)
  const int64_t first = gemma4::ArgmaxTextLogits(
      prefill.outputs[0], 4, prefill.seq_len, prefill.OutputCapacity(0));
  std::cout << "stage first token: " << first << std::endl;
  prefill.Clear();
  decode.Clear();
  hbDNNRelease(packed);

  // High-level session: the same stages orchestrated multi-turn.
  gemma4::TextEngine engine(argv[1], argv[2]);
  engine.SetDebugSink([](const std::string &m) { std::cerr << m << "\n"; });
  const auto out = engine.Generate(prompt, 2);
  const auto next = engine.ContinueGenerate(out, 1);
  std::cout << "session processed: " << engine.ProcessedTokens() << std::endl;
  engine.ResetSession();
  return 0;
}
```

The program prints the first generated token ID and the session token count. Generated IDs depend on the supplied model and prompt.

Lifetime rules for the example: subgraph handles are borrowed from the packed
model, so `prefill`/`decode` must be `Clear`ed before the packed model is
released; KV input slots borrow `KvCache` memory, which stays valid until the
cache is reallocated or destroyed; `CollectKvOutputs` rows borrow the output
tensors and are only valid until the next inference. A `TextEngine` (and each
stage operating on a `ModelIo`) is not thread-safe — serialize access; the
stage functions themselves hold no global state. Inference failures propagate
as exceptions with the task released and all buffers still owned, so a session
can `ResetSession` and continue.

Host test entry:
```bash
python3 -m unittest discover -s samples/llm/gemma4-e2b/tests -p test_cpp_text_stages.py -v
```
