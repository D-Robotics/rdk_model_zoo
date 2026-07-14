# Gemma4-E2B VLM Model Description

[简体中文](./README_cn.md) | **English**

<p align="center">
  <img src="./test_data/results/image.jpg" alt="Gemma4-E2B on RDK S100P" width="960">
</p>

Real-time **Vision-Language Model** inference for Google **Gemma4-E2B** on **D-Robotics RDK S100P** (`march=nash-m`). Runs fully on-device via the BPU — text chat and image+text (VLM) in one interactive runtime.

![Text chat demo](./test_data/results/test3.jpg)

*Text chat on S100P: Chinese prompt, BPU streaming output (~6.9 tok/s).*

![VLM demo](./test_data/results/test1.jpg)

*VLM chat: load an image, ask in Chinese, stream the reply (86% BPU utilization).*

> Supported platform: **RDK S100P** (community sample; upstream: [gemma4-e2b-rdk-s100p](https://github.com/shockley6668/gemma4-e2b-rdk-s100p))

---

## Algorithm Overview

Gemma4-E2B is a lightweight multimodal model from Google, combining a Vision ViT encoder with a 2B-parameter Text LLM decoder. Official materials:

- Model card: https://huggingface.co/google/gemma-4-e2b
- Upstream deployment project: https://github.com/shockley6668/gemma4-e2b-rdk-s100p

### Algorithm Capabilities

- Multimodal understanding: image + text → text
- Multi-turn text chat with KV cache reuse
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
| RDK S100 / S600 | ❌ | Not validated in this sample |

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
./run.sh
```

The script will:

1. Install build dependencies (`cmake`, `g++`, `libopencv-dev`, `libgflags-dev`, `nlohmann-json3-dev`, `cargo`, `wget`)
2. Download pre-compiled runtime model files from the D-Robotics model archive if missing (`~/gemma4_e2b` by default)
3. Download `tokenizers-cpp` source (pinned commit)
4. Build `main` (first build compiles `tokenizers-cpp`, ~few minutes)
5. Launch interactive VLM chat

Example session:

```
gemma4> /image ../../test_data/image1.jpg
gemma4> Describe this image
gemma4> /reset
gemma4> /quit
```

For step-by-step details see [runtime/cpp/README.md](./runtime/cpp/README.md).

---

## Model Conversion

ModelZoo already provides pre-compiled HBM models. Users can download them
directly via `model/download_model.sh` and **skip this section** if they
only want to run inference.

For custom re-quantization (requires a PC with 128 GB RAM + OE-LLM SDK),
see [conversion/README.md](./conversion/README.md) and the full guide:

- [QUANTIZATION_TUTORIAL.md](./conversion/QUANTIZATION_TUTORIAL.md) (English)
- [QUANTIZATION_TUTORIAL_zh.md](./conversion/QUANTIZATION_TUTORIAL_zh.md) (中文)

---

## Runtime

This sample provides a **C++** board-side runtime only (LLM inference is
C++-native; no Python path is provided). For build, parameters, and
detailed usage, see [runtime/cpp/README.md](./runtime/cpp/README.md).

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
