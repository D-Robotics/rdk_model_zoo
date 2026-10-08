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

> Screenshots and board performance figures below are records from the source S release.

---

<a id="overview"></a>
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

<a id="directory"></a>
## Directory structure

```text
gemma4-e2b/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python and native inference implementations
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── third_party/  # Files for third_party
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

---

<a id="support-matrix"></a>
## Platform Compatibility

| Platform | Support | Notes |
| --- | --- | --- |
| RDK S100P | ✅ | Primary target (`nash-m`, `core_num=1`) |
| RDK S600 | ✅ | `nash-p`; matching public S600 HBMs, with Vision and Text loaded once at startup and kept resident |
| RDK S100 | ⚠️ | SoC branch included; S100 HBMs are not published — supply them via `GEMMA4_MODEL_BASE_URL` (see [model preparation](model/README.md)); performance figures below were recorded on S600 |

S100 uses its SoC branch; supply matching S100 HBMs through `GEMMA4_MODEL_BASE_URL`
(see [model preparation](model/README.md)). The performance figures are S600
source-release records.

---

<a id="prerequisites"></a>
## Prerequisites

The board needs the matching OE-LLM runtime, C++17 build tools, OpenCV, gflags, JSON headers and Rust 1.80+.
The launcher additionally uses the Python 3 standard library. See [native dependencies](runtime/cpp/README.md#dependencies) for installation commands.
Prepare the two SoC-matched HBMs, shared embedding table and tokenizer. Keep targets in separate data directories because HBM filenames are identical.
See [model preparation](model/README.md) for sizes and source-recorded checksums. Inference does not require a PC quantization toolchain.

<a id="quickstart"></a>
## Quick Start

Install the [C++ prerequisites](runtime/cpp/README.md#dependencies) first. From the repository root, for S600:

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

Model download, dependency preparation, build and execution are separate steps. `run.sh` uses Python 3 for target detection and process launch; tokenization and inference remain C++. Launch never installs dependencies, downloads models or builds automatically. On a host, `./run.sh --target s600 --dry-run` prints the launch command without executing it.

Example session:

```
gemma4> /image ../../test_data/image1.jpg
gemma4> Describe this image
gemma4> /context
gemma4> /reset
gemma4> /quit
```

`main` defaults to `--max_tokens=0`, which uses all KV capacity remaining after the current prompt while keeping `prompt + output <= 4096`. With explicit `GEMMA4_SOC=s600`, `download_model.sh` selects the public `nash-p` Vision/Text HBMs automatically.

For step-by-step details see [runtime/cpp/README.md](./runtime/cpp/README.md).

---

## Model Conversion

Pre-compiled HBM models can be used directly, so users who only need
inference may **skip this section**. The download helper selects the public
S100P or S600 assets using the explicit `GEMMA4_SOC`; S100 requires matching HBMs supplied locally
or through `GEMMA4_MODEL_BASE_URL`.

For custom re-quantization (requires a PC with 128 GB RAM + OE-LLM SDK),
see [conversion/README.md](./conversion/README.md) and the full guide:

- [QUANTIZATION_TUTORIAL.md](./conversion/QUANTIZATION_TUTORIAL.md) (English)
- [QUANTIZATION_TUTORIAL_zh.md](./conversion/QUANTIZATION_TUTORIAL_zh.md) (中文)

---

<a id="expected-results"></a>
## Inference Result

![VLM demo](./test_data/results/test1.jpg)

*VLM chat on S100P: image + Chinese prompt → streamed BPU reply.*

---

<a id="entry-points"></a>
## Runtime

This sample provides a **C++** board-side runtime only (LLM inference is
C++-native; no Python path is provided). For build, parameters, and
interactive / OpenAI-compatible chat usage, see [runtime/cpp/README.md](./runtime/cpp/README.md).

---

## Model Evaluation

The `evaluator/` directory documents accuracy / golden-tensor verification.
See [evaluator/README.md](./evaluator/README.md).

---

<a id="license"></a>
## License

Runtime C++ code in this sample is MIT-licensed (see upstream
[gemma4-e2b-rdk-s100p](https://github.com/shockley6668/gemma4-e2b-rdk-s100p)).
Pre-compiled models are distributed separately through the D-Robotics model archive. The sample
itself follows the Model Zoo top-level License.
