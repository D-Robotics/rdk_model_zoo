# Model Conversion (PC-side)

[简体中文](./README_cn.md) | **English**

PTQ quantization and HBM compilation run on a **development PC**, not on the board.

## Requirements

| Item | Minimum | Recommended |
| --- | --- | --- |
| RAM | 64 GB | 128 GB+ (Text compile peaks ~100 GB) |
| OS | Ubuntu 22.04 | Same |
| SDK | OE-LLM 1.0.0 | D-Robotics official channel |
| GPU | Optional | CUDA for Vision calibration |

## Contents

```bash
conversion/
├── leap_llm_gemma4/          Gemma4 model defs for leap_llm
│   ├── models/gemma4/
│   └── apis/model/
└── scripts/
    ├── calibration/          COCO image + text prompt prep
    ├── compile/              Vision/Text HBM compile scripts
    └── verify/               BC/HBM accuracy verification
```

## Quick Reference

```bash
# Vision compile (GPU recommended)
bash conversion/scripts/compile/run_vision_compile.sh

# Text compile (CPU, ~100 GB RAM peak)
bash conversion/scripts/compile/run_text_compile.sh

# Adjust context at compile time
CHUNK_SIZE=512 CACHE_LEN=8192 bash conversion/scripts/compile/run_text_compile.sh
```

Then sync `kChunkSize` / `kCacheLen` in `runtime/cpp/gemma4_config.h` and rebuild the board runtime.

## Full Tutorial

See the step-by-step guide with pitfalls and solutions:

- [QUANTIZATION_TUTORIAL.md](./QUANTIZATION_TUTORIAL.md) (English)
- [QUANTIZATION_TUTORIAL_zh.md](./QUANTIZATION_TUTORIAL_zh.md) (中文)

Install Gemma4 adapters into your OE-LLM environment:

```bash
bash conversion/leap_llm_gemma4/install.sh
```
