# C++ Runtime

[中文](./README_cn.md) | **English**

C++ inference runtime for Gemma4-E2B VLM on D-Robotics RDK S100P (`march=nash-m`). Loads pre-compiled HBM models and runs real-time Vision-Language inference on the BPU.

> Part of [Gemma4-E2B sample](../../README.md). Full upstream project: [gemma4-e2b-rdk-s100p](https://github.com/shockley6668/gemma4-e2b-rdk-s100p).

## Prerequisites

The S100P board must have the OE-LLM runtime installed:

```bash
# Verify Horizon BPU SDK
ls /usr/hobot/lib/libdnn.so    # BPU inference lib
ls /usr/hobot/lib/libhbucp.so  # Memory management lib
ls /usr/include/hobot/dnn/hb_dnn.h
```

System dependencies (usually pre-installed on OE-LLM images):

```bash
sudo apt install cmake g++ libopencv-dev libgflags-dev nlohmann-json3-dev cargo wget
```

> **No Python required.** Tokenization is done in native C++ via
> `tokenizers-cpp` (vendored in `third_party/`), matching the
> OpenExplorer_LLM-s600 reference implementation.

## Directory Layout

```
runtime/cpp/                            C++ source code (this directory)
├── CMakeLists.txt                      Build entry (pulls in tokenizers-cpp + gflags)
├── run.sh                              One-click build + interactive chat
├── inc/                                Public headers
│   ├── gemma4_config.hpp               Model constants (image token IDs, dims, ...)
│   ├── gemma4_text_engine.hpp          Text LLM engine (prefill + decode + KV cache)
│   ├── gemma4_vision_engine.hpp        Vision ViT engine
│   ├── gemma4_embeddings.hpp           Token embedding lookup + vision injection
│   ├── gemma4_kv_cache.hpp             Zero-copy KV cache management
│   ├── gemma4_vision_preprocess.hpp    Image resize + patchify
│   ├── gemma4_native_tokenizer.hpp     Native C++ tokenizer (from OE-LLM-s600)
│   ├── gemma4_tokenizer.hpp            TokenizerBridge: chat template + image expand
│   └── hb_utils.hpp                    Horizon BPU helpers (tensor, flush, infer)
└── src/                                Implementation + executables
    ├── main.cpp                        ★ Interactive VLM chat (primary entry)
    ├── gemma4_server.cpp               HTTP API server
    ├── gemma4_demo.cpp                 Single-shot VLM demo
    ├── gemma4_text_bench.cpp           Text-only benchmark
    ├── gemma4_golden_verify.cpp        Golden mask/KV alignment checker
    └── gemma4_*.cpp                    Engine implementations

../../third_party/
└── tokenizers-cpp/                     Downloaded at build time (see third_party/README.md)
```

## Build

Quick start (recommended):

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh
```

Manual build:

```bash
cd runtime/cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

The first build downloads and compiles `tokenizers-cpp` (HF tokenizers Rust
binding + sentencepiece + abseil), which takes a few minutes. Subsequent
builds are incremental and fast.

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
export GEMMA4_HOME=~/gemma4_e2b
bash ../../model/download_model.sh
```

This downloads the 3 runtime model files and the 2 required tokenizer files
from the D-Robotics model archive.

```
~/gemma4_e2b/
├── model/
│   ├── gemma4-e2b_vit_ptq.hbm                          # 329 MB  Vision
│   ├── gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm      # 4.5 GB  Text
│   └── tok_embeddings.bin                               # 1.5 GB  Embedding
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

## Run

Set `GEMMA4_HOME` to point at the model directory, then run:

```bash
export GEMMA4_HOME=~/gemma4_e2b

# Interactive VLM chat (zero-arg default uses $GEMMA4_HOME)
./main

# Inside the chat:
#   /image /path/to/photo.jpg        Load an image
#   What do you see in this image?   Ask a question
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
| `--max_tokens` | int | `4096` (`kCacheLen`) | Max new tokens per turn |

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

### `gemma4_server` — long-running chat server

| Flag | Type | Default | Description |
|---|---|---|---|
| `--text_hbm` | string | same as `main` | Text LLM HBM |
| `--vision_hbm` | string | same as `main` | Vision ViT HBM |
| `--tok_embeddings` | string | same as `main` | Token embedding table |
| `--max_tokens` | int | `128` | Max new tokens per request |

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

## Key Design Decisions

1. **Vision injection is raw** — ViT output `[280, 1536]` is injected directly into `inputs_embeds` at image soft-token positions (token ID 249560). No L2-norm scaling, no √1536 multiplication.

2. **PLE uses pad embedding** — At image positions, the Per-Layer Embedding token-identity path uses `pad_token_id=0` (not 249560), matching HuggingFace's `masked_scatter` behavior.

3. **Chat template** — Prompts are formatted in C++ to the Gemma turn format (`<bos><|turn>user\n...<turn|>\n<|turn>model\n`), matching `chat_template.jinja`. Tokenization uses the native `tokenizers-cpp` (HF tokenizers), not Python.

4. **Zero-copy KV cache** — KV cache memory is allocated once and shared between prefill and decode via pointer assignment, avoiding per-step memcpy.

5. **Chunked prefill** — Prompts longer than `chunk_size=256` tokens are automatically split into multiple prefill chunks.

## Verification

To verify board inference matches the PC golden data:

```bash
# Optional internal verification data: place golden_mask_kv/ under
# $GEMMA4_HOME/golden_mask_kv/. It is not included in the public model archive.
./gemma4_golden_verify --prompt_id prompt_0
# Expected: ALL PASSED (cosine=1.0 for all 5 tensors)
```
