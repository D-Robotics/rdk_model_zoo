# Model Conversion (PC-side)

[简体中文](./README_cn.md) | **English**

PTQ quantization and HBM compilation run on a **development PC**, not on the board.

<a id="source-model"></a>
## Source model and recipe

This directory carries the Gemma4-E2B conversion recipe from the source S release: original weights,
Gemma4 leap_llm adaptation, Vision/Text calibration, PTQ compilation and verification tools.
The model source is `google/gemma-4-e2b`; follow full tutorial §3.4 for weight acquisition and access requirements.
Commands here start in `samples/llm/gemma4-e2b` (sample root) unless a code block changes directory explicitly.
The quantization workflow is the tutorial's own; run it in the OE environment when preparing artifacts.

<a id="toolchain-targets"></a>
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

<a id="export"></a>
## Model adaptation and export entry

Install the Gemma4 adaptation into an already prepared OE-LLM environment:

```bash
bash conversion/leap_llm_gemma4/install.sh
```

Graph export is handled by the OE-LLM model adaptation and compiler entry points; this recipe has no separate ONNX export step.
See full tutorial §2, §5 and §6 for architecture and Vision/Text interfaces; do not substitute another sample's ONNX commands.

<a id="calibration"></a>
## Calibration data

```bash
# Prepare exactly 50 deterministic real COCO val2017 images.
python3 conversion/scripts/calibration/download_coco_images.py
```

<a id="compile"></a>
## Compilation

```bash
# Select one target; use s100 or s100p for the other boards.
TARGET_SOC=s600 bash conversion/scripts/compile/run_vision_compile.sh
TARGET_SOC=s600 bash conversion/scripts/compile/run_text_compile.sh
```

The Vision script refuses synthetic or untracked images: the calibration
directory must match `images_coco_manifest.json`. Text compilation uses the
existing text calibration corpus and does not generate replacement prompts.

The released Text HBM is compiled with `CHUNK_SIZE=256` and
`CACHE_LEN=4096`; no 8K/16K HBM is published. The
interactive `main` binary uses all KV capacity left after the prompt when
`--max_tokens=0` (the default), so no HBM rebuild is needed to maximize the
current 4096-token budget.

If a different HBM is compiled later, keep `kChunkSize` / `kCacheLen` in
`runtime/cpp/inc/gemma4_config.hpp` exactly synchronized with its compile-time
settings before rebuilding the board runtime.

## Full Tutorial

The full tutorial walks through the quantization workflow step by step. For board startup/build instructions, use the current [C++ README](../runtime/cpp/README.md#build), which separates preparation, build and run.

See the step-by-step guide with pitfalls and solutions:

- [QUANTIZATION_TUTORIAL.md](./QUANTIZATION_TUTORIAL.md) (English)
- [QUANTIZATION_TUTORIAL_zh.md](./QUANTIZATION_TUTORIAL_zh.md) (中文)

<a id="validation"></a>
## Verification workflow

The [evaluator guide](../evaluator/README.md) preserves PC BC/float comparison and board golden-input alignment commands.
The full tutorial explains accuracy observations and troubleshooting; those workflows are executed by the user in the OE environment.

<a id="artifacts"></a>
## Produced artifacts

Vision and Text use `conversion/output/gemma4_e2b_vision_<target>/` and
`conversion/output/gemma4_e2b_text_<target>/`. Board inference uses `gemma4-e2b_vit_ptq.hbm` and
`gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm`, plus embedding and tokenizer files.
See [model preparation](../model/README.md) for layout and source checksum records.
A `.bc` is a PC verification artifact, not a board HBM.

<a id="known-gaps"></a>
## Usage boundaries

The original prerequisites remain: weights, SDK and text calibration corpus must be prepared; Vision uses real images with a manifest.
S100P and S600 publish HBMs; on S100, supply the HBM artifacts yourself (see [model](../model/README.md)).
The recipe produces the documented 4096-token Vision/Text HBMs. For other context sizes, recompile with matching `CACHE_LEN` and synchronize `kChunkSize`/`kCacheLen` before rebuilding the runtime; see [Compilation](../runtime/cpp/README.md#build).
