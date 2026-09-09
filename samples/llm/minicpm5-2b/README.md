[English](README.md) | [简体中文](README_cn.md)

# MiniCPM5-2B on RDK S100 / S100P / S600

This sample runs text generation with OpenBMB MiniCPM5-2B using the S600 BPU and OELLM Runtime. It provides a C++ command-line application, model download with SHA256 verification, conversion guidance and full WikiText2 evaluation evidence.

> S600 currently requires the internal OELLM 2.0 beta SDK; its public release is planned for mid-October 2026. Public S600 SDK 1.0.5 did not pass direct inference with this HBM and cannot substitute for 2.0. S100/S100P use the separate public 1.0.0 workflow below.

## S100 / S100P support

S100 (Nash-e) and S100P (Nash-m) use separate OELLM 1.0.0 W8 artifacts and the [legacy C++ entry point](runtime/legacy/README.md). Both pass single-turn Chinese/English generation and normal EOS; full PPL is unverified. Short-request decode is approximately 12.1 / 13.0 tokens/s. Follow that entry point for memory configuration, downloads and commands; see [legacy conversion](conversion/legacy/README.md). The existing PPL, multi-turn and stability results below apply only to S600.


## S600 model and supported scope

MiniCPM5-2B uses a Llama architecture with 42 layers, hidden size 2048, 16 query heads and 2 KV heads. This artifact uses a 256-token prefill chunk, a 4096-token KV cache and four Nash-p cores. Generation is greedy with thinking disabled. It supports Chinese/English text and a follow-up prompt in the same conversation.

This delivery is verified on **S600** with RDK OS V5.1.0. The artifact is not interchangeable with S100/S100P models. The original model's longer context limit does not apply to this compiled 4096-token configuration. Image input, tool execution and a serving API are outside this sample's scope.

## Directory layout

```text
conversion/     Host adapter and quantization/compilation instructions
evaluator/      Full PPL evaluator and recorded evidence
model/          Verified model download
runtime/cpp/    S600 CMake project and run.sh
runtime/legacy/ S100/S100P CMake project and run.sh
test_data/      Generation prompts and observed reference results
```

## S600 quick start

Obtain and extract OpenExplorer LLM 2.0.0-beta1, including its `oellm_runtime` directory. On the S600 board install the build dependencies:

```bash
sudo apt-get install build-essential cmake libgflags-dev nlohmann-json3-dev curl
export OELLM_SDK_ROOT=/path/to/OpenExplorer_LLM
cd samples/llm/minicpm5-2b/runtime/cpp
bash run.sh
bash run.sh --prompt="What is the capital of France? Answer with the city name only." \
  --follow_up="Translate the previous answer into Chinese. Answer with the city name only."
```

Allow at least 6 GB free storage for archive download and extraction. The SDK is obtained separately; its libraries and headers are not included in the model download. See [model download](model/README.md) and [runtime options](runtime/cpp/README.md).

## Conversion and evaluation

See [conversion](conversion/README.md) for the external MiniCPM5 adapter and the pinned SDK environment. Do not edit the SDK's `deps_version.conf`. See [evaluation](evaluator/README.md) for data preparation, complete PPL statistics and measurement conditions.

| Full WikiText2 TEST, 140 × 2048 tokens | PPL |
| --- | ---: |
| Floating-point reference | 14.0184 |
| Fake-quantized model | 14.2687 |
| Final S600 HBM | 14.2428 |

The final HBM increases PPL by 1.60% relative to the floating-point reference. All 286580 next-token predictions were evaluated. The board-local NumPy evaluator differs from the SDK/PyTorch RPC path by 0.0055% PPL on a five-sample cross-check; it is not bit-identical.

Observed generation tests include six additional prompts matching reference text and tokens, two-turn English/Chinese conversation, retrieval from inputs padded to 2048 and 3840 tokens, and 50 repeated requests. The latter averaged 53.25 decode tokens/s and 147.86 ms time to first token for the recorded short prompt. These are workload-specific measurements, not concurrency or long-duration stability guarantees. Runtime prefill token counts are padded to chunk boundaries.

The reference model itself answered one Chinese `1+1` prompt incorrectly and added Markdown fences when asked for bare JSON. Quantization preserved those responses; this sample does not promise strict JSON formatting or universal factual correctness.

## License and attribution

Original model: [OpenBMB/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B), revision `0e9c66dce9fedde5ba8663bbcdd54b6810bb929a`, Apache-2.0. The model archive includes LICENSE and a notice describing quantization and metadata changes. Sample code follows the repository license. OpenExplorer/OELLM remains subject to its separate SDK terms.
