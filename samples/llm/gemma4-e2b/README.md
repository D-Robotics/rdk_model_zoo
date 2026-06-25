# Gemma4-E2B VLM

[简体中文](./README_cn.md) | **English**

<p align="center">
  <img src="./docs/image.jpg" alt="Gemma4-E2B on RDK S100P" width="960">
</p>

Real-time **Vision-Language Model** inference for Google **Gemma4-E2B** on **D-Robotics RDK S100P** (`march=nash-m`). Runs fully on-device via the BPU — text chat and image+text (VLM) in one interactive runtime.

![Text chat demo](./docs/test3.jpg)

*Text chat on S100P: Chinese prompt, BPU streaming output (~6.9 tok/s).*

![VLM demo](./docs/test1.jpg)

*VLM chat: load an image, ask in Chinese, stream the reply (86% BPU utilization).*

> Supported platform: **RDK S100P** (community sample; upstream: [gemma4-e2b-rdk-s100p](https://github.com/shockley6668/gemma4-e2b-rdk-s100p))

---

## Algorithm Overview

Gemma4-E2B is a lightweight multimodal model (Vision ViT + 2B Text LLM):

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
├── conversion/                  PC-side PTQ compile (128 GB RAM + OE-LLM SDK)
│   ├── leap_llm_gemma4/
│   ├── scripts/
│   └── README.md
├── runtime/
│   └── cpp/                     ★ Board-side C++ inference (gemma4_chat)
│       ├── run.sh
│       └── README.md
├── evaluator/                   Accuracy / golden verification
│   └── README.md
├── docs/                        Full quantization & deployment tutorial
│   ├── QUANTIZATION_TUTORIAL.md
│   └── QUANTIZATION_TUTORIAL_zh.md
├── test_data/                   VLM test images (red panda, etc.)
└── third_party/
    └── tokenizers-cpp/          Vendored HF tokenizers C++ binding
```

---

## Quick Start

Each sample provides a `run.sh` for one-click experience on board:

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh
```

The script will:

1. Install build dependencies (`cmake`, `g++`, `libopencv-dev`, `cargo`)
2. Download pre-compiled models from HuggingFace if missing (`~/gemma4_e2b` by default)
3. Build `gemma4_chat` (first build compiles `tokenizers-cpp`, ~few minutes)
4. Launch interactive VLM chat

Example session:

```
gemma4> /image ../../test_data/image1.jpg
gemma4> Describe this image
gemma4> /reset
gemma4> /quit
```

For step-by-step details see [runtime/cpp/README.md](./runtime/cpp/README.md).

---

## Re-quantize from Source

Requires a PC with **128 GB RAM** and the **OE-LLM SDK**. Full guide:

- [QUANTIZATION_TUTORIAL.md](./docs/QUANTIZATION_TUTORIAL.md) (English)
- [QUANTIZATION_TUTORIAL_zh.md](./docs/QUANTIZATION_TUTORIAL_zh.md) (中文)

---

## License Note

Runtime C++ code in this sample is MIT-licensed (see upstream [gemma4-e2b-rdk-s100p](https://github.com/shockley6668/gemma4-e2b-rdk-s100p)). Pre-compiled models are distributed separately on HuggingFace.
