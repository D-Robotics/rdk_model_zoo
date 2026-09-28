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

## Target SoCs

The same conversion entry points support all three RDK S targets. `TARGET_SOC`
defaults to `s100p` for backward compatibility.

| `TARGET_SOC` | HBDK march | Vision cores | Text prefill / decode cores |
| --- | --- | ---: | ---: |
| `s100` | `nash-e` | 1 | 1 / 1 |
| `s100p` | `nash-m` | 1 | 1 / 1 |
| `s600` | `nash-p` | 4 | 2 / 2 |

S600-specific dynamic quantization, `opt=1`, HPC and decode no-padding options
are enabled only for `nash-p`; S100/S100P retain the original single-core
behavior. The tested S600 board has 23 GiB usable RAM (24 GB nominal), not
64 GB.

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
# Prepare exactly 50 deterministic real COCO val2017 images.
python3 conversion/scripts/calibration/download_coco_images.py

# Select one target; use s100 or s100p for the other boards.
TARGET_SOC=s600 bash conversion/scripts/compile/run_vision_compile.sh
TARGET_SOC=s600 bash conversion/scripts/compile/run_text_compile.sh
```

The Vision script refuses synthetic or untracked images: the calibration
directory must match `images_coco_manifest.json`. Text compilation uses the
existing text calibration corpus and does not generate replacement prompts.

The released Text HBM is compiled with `CHUNK_SIZE=256` and
`CACHE_LEN=4096`. This deliverable does not include an 8K/16K HBM. The
interactive `main` binary uses all KV capacity left after the prompt when
`--max_tokens=0` (the default), so no HBM rebuild is needed to maximize the
current 4096-token budget.

If a different HBM is compiled later, keep `kChunkSize` / `kCacheLen` in
`runtime/cpp/inc/gemma4_config.hpp` exactly synchronized with its compile-time
settings before rebuilding the board runtime.

## Full Tutorial

See the step-by-step guide with pitfalls and solutions:

- [QUANTIZATION_TUTORIAL.md](./QUANTIZATION_TUTORIAL.md) (English)
- [QUANTIZATION_TUTORIAL_zh.md](./QUANTIZATION_TUTORIAL_zh.md) (中文)

Install Gemma4 adapters into your OE-LLM environment:

```bash
bash conversion/leap_llm_gemma4/install.sh
```
