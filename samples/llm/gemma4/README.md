English | [简体中文](README_cn.md)

# Gemma4 E2B / E4B text generation

[简体中文](README_cn.md) · [Models](model/README.md) · [Model provenance](conversion/README.md) · [Runtime](runtime/cpp/README.md) · [Benchmarks](evaluator/README.md)

<a id="overview"></a>
## Model and source

This sample uses the official OELLM SDK 2.0.0 S600 packages and the `OellmRuntime` basic API.
E2B and E4B each provide W4 and W8 profiles. Single prompts, full-history follow-ups and interactive text chat are supported.
The SDK owns tokenization, both embedding tables, prefill/decode, KV cache and sampling.

<a id="directory"></a>
## Directory

```text
gemma4/
├── model/          # Four official profiles and preparation
├── runtime/cpp/    # OellmRuntime basic API
├── conversion/     # Vendor artifact provenance
├── evaluator/      # Vendor references and separate board records
├── test_data/      # Text prompts
└── tests/          # Host preparation checks
```

<a id="support-matrix"></a>
## Models and platform

All profiles below use the S600/Linux C++ entrypoint.

| Profile | ZIP | Bytes |
| --- | --- | ---: |
| `e2b-w4` | [Gemma4-E2B_w4.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E2B_w4.zip) | 4,402,995,725 |
| `e2b-w8` | [Gemma4-E2B_w8.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E2B_w8.zip) | 6,547,749,326 |
| `e4b-w4` | [Gemma4-E4B_w4.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E4B_w4.zip) | 6,320,670,645 |
| `e4b-w8` | [Gemma4-E4B_w8.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E4B_w8.zip) | 9,762,116,686 |

All profiles target S600/nash-p with four BPU cores, chunk 512 and context 4096.
W4/W8 labels describe the vendor language model; vision/audio weights are W8 and external embeddings are FP16.

<a id="prerequisites"></a>
## Prerequisites

On S600, prepare the matching `oellm_runtime_s600`, CMake/C++17 and Python 3 standard library.
Allow disk space for both the ZIP and extracted files; package sizes are listed above.

<a id="quickstart"></a>
## Quick start

Run from the repository root; this example selects E2B W4:

```bash
mkdir -p /data/gemma4-official
curl -fL --retry 3 -o Gemma4-E2B_w4.zip https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E2B_w4.zip
printf '%s  %s\n' 'bd9b1406e2cf4a6cde7cda769fbcbbf43d1fa9a797f59936def6f7d6dbf793d2' 'Gemma4-E2B_w4.zip' | sha256sum -c -
python3 -m zipfile -e Gemma4-E2B_w4.zip /data/gemma4-official
python3 samples/llm/gemma4/model/prepare.py \
  --profile e2b-w4 \
  --source-dir /data/gemma4-official/Gemma4-E2B_w4 \
  --output-dir /data/gemma4-official/Gemma4-E2B_w4

export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600
bash samples/llm/gemma4/runtime/cpp/run.sh \
  --config_path /data/gemma4-official/Gemma4-E2B_w4/oellm_config.json \
  --prompt "请把 hello 翻译成中文，只回答译文。" --max_new_tokens 64
```

Select another official ZIP and change `--profile` for the other profiles.
Full package and member digests are in `model/manifests/<profile>.json`.
Use `--interactive` for terminal chat, `/reset` to clear history, and `/quit` to exit.

<a id="expected-results"></a>
## Output and performance reference

A successful request emits `RESULT {JSON}` containing generated text, token IDs, completion status and available Runtime metrics.



These values are from the [official S600 benchmark](https://developer.d-robotics.cc/oe_llm_doc/guide/benchmark/s600/gemma_benchmark.html); TTFT uses its text-only column.

| Profile | TTFT (ms) | Decode (token/s) | Memory (GB) |
| --- | ---: | ---: | ---: |
| e2b-w4 | 74 | 61.2 | 9.9 |
| e2b-w8 | 79 | 47.8 | 13.5 |
| e4b-w4 | 137 | 38.3 | 15.4 |
| e4b-w8 | 139 | 27.9 | 15.5 |

Conditions: S600/Linux, four BPU cores, chunk 512, context 4096, standard single-request generation.
Memory is the vendor package's Linux+ION total, not text-only peak memory.
The reference does not publish a binary-SHA binding, PPL or two-request concurrency measurements.
Local board records remain `not-run` with null values in `evaluator/board_results/`.

<a id="entry-points"></a>
## Entrypoints and host checks

[Preparation](model/README.md), [inference](runtime/cpp/README.md), [artifact provenance](conversion/README.md) and [evaluation](evaluator/README.md).

```bash
python3 -m unittest discover -s samples/llm/gemma4/tests
```

Preparation and inference require neither a Skill, Node nor a generated website catalog, and install no dependencies.
The existing `develop` sample at `samples/llm/gemma4-e2b` is a separate native C++ implementation.

<a id="license"></a>
## Provenance and licensing

The official package SHA256, member digests, actual compiler and Runtime configuration are recorded independently.
The ZIP supplies no full upstream HF revision; it remains null instead of inheriting another conversion's revision.
Model: [Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license). Sample: [Apache-2.0](LICENSE).
Runtime libraries and headers retain their SDK terms and are obtained separately.
