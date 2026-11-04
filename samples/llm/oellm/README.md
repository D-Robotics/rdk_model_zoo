English | [简体中文](README_cn.md)

# Official OELLM S600 models

<a id="overview"></a>
## Models and source

Use the official OELLM SDK 2.0.0 S600 ZIPs and matching Runtime for 32 text, vision-language, ASR and Eagle3/DFlash profiles. Downloads go directly to vendor OSS. Existing Gemma 4 profiles remain in their [separate sample](../gemma4/README.md).

<a id="directory"></a>
## Directory

```text
oellm/
├── model/          # Official manifests and preparation
├── runtime/cpp/    # SDK C++ launchers and configurations
├── conversion/     # Precompiled provenance
├── evaluator/      # Vendor references and separate local records
├── test_data/      # Prompts and SDK media paths
└── tests/          # Host checks
```

<a id="support-matrix"></a>
## Models and platform

| Series | Profiles | Task |
| --- | ---: | --- |
| InternVL2 | 1 | vlm |
| InternVL2.5 | 1 | vlm |
| InternVL3.5 | 1 | vlm |
| MiniCPM5 | 1 | text-gen |
| Phi-4 Mini | 1 | text-gen |
| Qwen2.5 | 1 | text-gen |
| Qwen2.5-VL | 2 | vlm |
| Qwen3 | 11 | text-gen |
| Qwen3.5 | 3 | vlm |
| Qwen3-ASR | 2 | asr |
| Qwen3-VL | 6 | vlm |
| Whisper | 2 | asr |

All profiles target S600/Linux aarch64. Standard and accelerated profiles use their own complete ZIPs and configurations. Qwen3.5 4B language/vision graphs use one core; other language graphs use four. DFlash draft graphs separately use one core.

<a id="prerequisites"></a>
## Prerequisites

Prepare the complete SDK 2.0.0 `oellm_runtime_s600`, including `env.sh`, `lib/`, `include/`, `samples/basic/basic_sample`, `examples/asr_demo/asr_demo` and `resource/`. Launchers use the Python 3 standard library and Bash. Optional `--build` requires CMake 3.15+ and C++17. See the [model inventory](model/README.md) for disk requirements.

<a id="quickstart"></a>
## Quick start

From the repository root on S600, using Qwen3 0.6B W8:

```bash
curl -fL --retry 3 -o Qwen3-0.6B_w8.zip https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Qwen3-0.6B_w8.zip
printf '%s  %s\n' '54d95f45185d61ae284d270f1341f4d533dc68ce4263c78067f14c0ca5725327' 'Qwen3-0.6B_w8.zip' | sha256sum -c -
python3 -m zipfile -e Qwen3-0.6B_w8.zip /data
MODEL_DIR=/data/Qwen3-0.6B_w8
python3 samples/llm/oellm/model/prepare.py --profile qwen3-0-6b-w8 --source-dir "$MODEL_DIR"
export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600
bash samples/llm/oellm/runtime/cpp/run.sh --config_path "$MODEL_DIR/oellm_config.json" --prompt "你好"
```

List other profiles:

```bash
python3 samples/llm/oellm/model/prepare.py --list
```

For other profiles, download their ZIP and change the profile/extraction directory. See [Runtime](runtime/cpp/README.md#run) for image and audio commands.

<a id="expected-results"></a>
## Output and references

The SDK `basic_sample` prints `text_result` and Runtime metrics for text/images. SDK `asr_demo` prints `[Transcription]` and `[Metric]`. Website values come from vendor benchmarks; published speed ranges remain ranges and are separate from local board records.

<a id="entry-points"></a>
## Entrypoints

[model](model/README.md) · [runtime](runtime/cpp/README.md) · [conversion](conversion/README.md) · [evaluator](evaluator/README.md)

```bash
python3 -m unittest discover -s samples/llm/oellm/tests
```

Model preparation and native inference require no Skill, Node or generated website catalog and do not install components.

<a id="license"></a>
## Provenance and licenses

Sample scripts use [Apache-2.0](LICENSE). Weight licenses are recorded per profile in `source.license`: Qwen2.5 3B and Qwen2.5-VL 3B use the Qwen RESEARCH LICENSE AGREEMENT; Phi-4 Mini and InternVL2/2.5 use MIT; the remaining base models in this batch use Apache-2.0. Vendor ZIPs disclose no full HF revision, so it remains null. SDK C++ programs, headers and libraries retain their supplied SDK terms and are obtained separately.
