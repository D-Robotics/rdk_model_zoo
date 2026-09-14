# Gemma4-E2B VLM Model Description

[简体中文](./README_cn.md) | **English**

<p align="center">
  <img src="./test_data/results/image.jpg" alt="Gemma4-E2B on RDK S100P" width="960">
</p>

Real-time **Vision-Language Model** inference for Google **Gemma4-E2B** on **D-Robotics RDK S100P and S600**. It runs fully on-device via the BPU and provides multi-turn text chat plus image+text VLM chat in one interactive runtime.

![Text chat demo](./test_data/results/test3.jpg)

*Text chat on S100P: Chinese prompt, BPU streaming output (~6.9 tok/s).*

![VLM demo](./test_data/results/test1.jpg)

*VLM chat: load an image, ask in Chinese, stream the reply (86% BPU utilization).*

> Supported platforms: **RDK S100P / S600**. Both use the same C++ runtime, but each board requires HBM files compiled for its SoC.

---

## Algorithm Overview

Gemma4-E2B is a lightweight multimodal model from Google, combining a Vision ViT encoder with a 2B-parameter Text LLM decoder. Official materials:

- Model card: https://huggingface.co/google/gemma-4-e2b
- Upstream deployment project: https://github.com/shockley6668/gemma4-e2b-rdk-s100p

### Algorithm Capabilities

- Multimodal understanding: image + text → text
- Multi-turn text chat with KV cache reuse
- Automatic budgeting across the full 4096-token context, with `/context` reporting and complete-turn history trimming
- Streaming token output on BPU

### Algorithm Features

- **Vision**: 16-layer ViT → 280 soft tokens per image
- **Text**: 35-layer decoder with PLE + KV cache (4096 context)
- **Deployment**: Two HBMs (Vision + Text) + external `tok_embeddings.bin`
- **On-board runtime**: Native C++ (`tokenizers-cpp`), no Python at inference time

---

## Platform Compatibility

| Platform | Support | Notes |
| --- | --- | --- |
| RDK S100P | ✅ | Primary target (`nash-m`, `core_num=1`) |
| RDK S600 | ✅ | `nash-p`; matching public S600 HBMs, with Vision and Text loaded once at startup and kept resident |
| RDK S100 | ⚠️ | SoC runtime branch retained, but board validation is not complete |

On-board regression for this update was run only on RDK S600. S100/S100P
coverage is limited to shared-source target-matrix and compatibility checks;
no S100 board connection is required or claimed.

---

## Directory Structure

```bash
samples/llm/gemma4-e2b/
├── README.md / README_cn.md     Sample overview (this file)
├── model/                       Pre-compiled HBM download
│   ├── download_model.sh
│   └── README.md
├── conversion/                  PC-side PTQ compile and quantization tutorial
│   ├── QUANTIZATION_TUTORIAL.md
│   ├── QUANTIZATION_TUTORIAL_zh.md
│   ├── leap_llm_gemma4/
│   ├── scripts/
│   └── README.md
├── runtime/
│   └── cpp/                     ★ Board-side C++ inference (main)
│       ├── run.sh
│       └── README.md
├── evaluator/                   Accuracy / golden verification
│   └── README.md
├── test_data/                   VLM test images and result screenshots
│   └── results/
└── third_party/                 tokenizers-cpp (downloaded at build time)
    ├── install_tokenizers_cpp.sh
    └── README.md
```

---

## Quick Start

Each sample provides a `run.sh` for one-click experience on board:

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh                              # interactive VLM chat
./run.sh server --port=8000           # OpenAI / ChatBox API
./run.sh --max_tokens=512             # pass gflags to main
```

The script will:

1. Install build dependencies (`cmake`, `g++`, `libopencv-dev`, `libgflags-dev`, `nlohmann-json3-dev`, `cargo`, `wget`, `git`, `curl`)
2. Resolve the SoC-specific model files under `~/gemma4_e2b`: S100P and S600 download matching HBMs from the public archive; S100 uses pre-placed HBMs or `GEMMA4_MODEL_BASE_URL`
3. Download `tokenizers-cpp` source (pinned commit)
4. Build all five runtime targets (the first build compiles `tokenizers-cpp`)
5. Launch the selected entry; zero arguments still start interactive `main`

Example session:

```
gemma4> /image ../../test_data/image1.jpg
gemma4> Describe this image
gemma4> /context
gemma4> /reset
gemma4> /quit
```

`main` defaults to `--max_tokens=0`, which uses all KV capacity remaining after the current prompt while keeping `prompt + output <= 4096`. On S600, `download_model.sh` selects the public `nash-p` Vision/Text HBMs automatically.

For step-by-step details see [runtime/cpp/README.md](./runtime/cpp/README.md).

---

## Model Conversion

Pre-compiled HBM models can be used directly, so users who only need
inference may **skip this section**. The download helper selects the public
S100P and S600 assets automatically; S100 requires matching HBMs supplied locally
or through `GEMMA4_MODEL_BASE_URL`.

For custom re-quantization (requires a PC with 128 GB RAM + OE-LLM SDK),
see [conversion/README.md](./conversion/README.md) and the full guide:

- [QUANTIZATION_TUTORIAL.md](./conversion/QUANTIZATION_TUTORIAL.md) (English)
- [QUANTIZATION_TUTORIAL_zh.md](./conversion/QUANTIZATION_TUTORIAL_zh.md) (中文)

---

## Runtime

This sample provides a **C++** board-side runtime only (LLM inference is
C++-native; no Python path is provided). For build, parameters, and
interactive / OpenAI-compatible chat usage, see [runtime/cpp/README.md](./runtime/cpp/README.md).

---

## Model Evaluation

The `evaluator/` directory documents accuracy / golden-tensor verification.
See [evaluator/README.md](./evaluator/README.md).

---

## Inference Result

![VLM demo](./test_data/results/test1.jpg)

*VLM chat on S100P: image + Chinese prompt → streamed BPU reply.*

---

## License

Runtime C++ code in this sample is MIT-licensed (see upstream
[gemma4-e2b-rdk-s100p](https://github.com/shockley6668/gemma4-e2b-rdk-s100p)).
Pre-compiled models are distributed separately through the D-Robotics model archive. The sample
itself follows the Model Zoo top-level License.
