# Gemma4-E2B text generation

[简体中文](README_cn.md) · [Model preparation](model/README.md) · [Conversion](conversion/README.md) · [Runtime](runtime/cpp/README.md) · [Evaluation](evaluator/README.md)

This sample integrates **the official OELLM SDK 2.0.0 `OellmRuntime`** for S600.
It provides single prompts, follow-up questions and an interactive text shell.
The SDK owns tokenization, PLE embeddings, prefill/decode, KV state and sampling.

| Item | Recorded configuration |
| --- | --- |
| Source | `google/gemma-4-E2B-it-qat-q4_0-unquantized` |
| Revision | `6befbaca7398925921802abd1f277b495b78b738` |
| Conversion | SDK AutoClip → GPTQ → HPP → HBDK; S600 HBM generated |
| Precision | Mixed W4/W8: Q/O/MLP etc. W4, K/V projections W8; SDK A8 policy |
| Target | `nash-p`, 4 cores, prefill chunk 512, total context 4096 |
| Board accuracy, performance, memory | `not-run`; values remain null |
| Artifacts | Complete Archive package; sizes and SHA256 recorded |
| Public model download | [S600 weights, 4.76 GB](https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/rdk_s600/gemma/gemma4-e2b-w4w8-s600-chunk512-ctx4096.tar.gz) |

This version supplies an S600 profile. The historical OE-LLM 1.0 sample under
`platforms/s/samples/llm/gemma4-e2b` retains its separate model/runtime pairing.

## Quickstart

Download and extract the Archive package as described in [model preparation](model/README.md),
then run these commands from the repository root:

```bash
python3 samples/llm/gemma4/model/prepare.py \
  --source-dir /data/gemma4-e2b-sdk2 \
  --output-dir /data/gemma4-e2b-sdk2

# Run on S600 with the matching SDK deployment package and CMake/C++17 tools.
export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600
bash samples/llm/gemma4/runtime/cpp/run.sh \
  --config_path /data/gemma4-e2b-sdk2/oellm_config.json \
  --prompt "What is 2 + 2? Answer with the number only." --max_new_tokens 64
```

SDK 2.0.0's GEMMA4 initialization requires a vision HBM even for text requests.
The sample checks this official companion during preparation. Its CLI accepts
text only. The complete set of files is approximately 7.30 GB; runtime memory
has not been measured.

Use `--interactive` for a text shell, `/reset` to clear client history and
`/quit` to exit. Each request supplies the full history. Template, history,
prompt and output share the compiled 4096-token context budget.

## Source and validation

`model/artifacts.json` binds source, target, compiler and file hashes.
`model/prepare.py` performs explicit preparation. `runtime/cpp/s600.json` derives
from the SDK's Gemma4 configuration; `gemma4.cc` calls its basic API.
Inference requires neither Skills, Node nor a generated Web catalog. The launch
script does not install dependencies or download models.

```bash
python3 -m unittest discover -s samples/llm/gemma4/tests
```

Host quantization references and board status are kept separate in
[evaluation](evaluator/README.md). `evaluator/board_result.json` remains
`not-run` with null measurements.

## License

The source model uses [Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license);
its original model card accompanies the artifacts. Sample code uses
[Apache-2.0](LICENSE), consistent with the S-series samples. SDK libraries and headers retain their own terms and must be obtained separately.
The bundled official companion preserves its separate SDK origin and terms.
