# Model preparation

[简体中文](README_cn.md) · [Sample](../README.md)

`artifacts.json` binds the SDK 2.0 S600 text deployment to its source revision,
compiler and complete file hashes. The language artifacts come from this run's
AutoClip/GPTQ → HPP → HBDK conversion. The generic SDK `_w8_` filename does not
describe the actual mixed W4/W8 projection policy.

## Download the Archive package

The [S600 deployment package](https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/rdk_s600/gemma/gemma4-e2b-w4w8-s600-chunk512-ctx4096.tar.gz) is 4,764,575,933 bytes compressed. It includes all eight recorded model files, the configuration template, model card and licenses.
From the repository root, download, verify and prepare it explicitly:

```bash
curl -fL --retry 3 -o gemma4-e2b-w4w8-s600-chunk512-ctx4096.tar.gz \
  https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/rdk_s600/gemma/gemma4-e2b-w4w8-s600-chunk512-ctx4096.tar.gz
printf '%s  %s\n' '619696547bb487c11d27a81e3bbdcf5358a93fbffefda239d6329900a57bd2d9' 'gemma4-e2b-w4w8-s600-chunk512-ctx4096.tar.gz' | sha256sum -c -
mkdir -p /data/gemma4-e2b-sdk2
tar -xzf gemma4-e2b-w4w8-s600-chunk512-ctx4096.tar.gz --strip-components=1 -C /data/gemma4-e2b-sdk2
python3 samples/llm/gemma4/model/prepare.py \
  --source-dir /data/gemma4-e2b-sdk2 \
  --output-dir /data/gemma4-e2b-sdk2
```

## Package layout

```text
gemma4-e2b-w4w8-s600-chunk512-ctx4096/
├── model_language_chunk_512_cache_4096_w8_nash-p_corenum_4_4.hbm
├── fake_quant_model_embed_tokens.bin
├── fake_quant_model_embed_tokens_per_layer.bin
├── gemma-4-E2B-it_vision_384x384_w8_nash-p_corenum_4.hbm
├── tokenizer.json
├── tokenizer_config.json
├── chat_template.jinja
├── oellm_config.template.json
├── model-manifest.json
├── SHA256SUMS
├── SOURCE_MODEL_README.md
├── README.md
└── LICENSE
```

The package contains 13 files. `prepare.py` generates `oellm_config.json` for the actual directory. Official runtime libraries and sample source are supplied separately.

## Prepare existing artifacts

The toolchain run directory is:

```text
/opt/D-Robotics_LLM_2.0.0_S600_S100_SDK/adapt_runs/gemma4_e2b_s600_20261103_ede84003/
  artifacts/hpp/hbm/              language artifacts and tokenizer/template
  artifacts/runtime_companions/   official SDK vision HBM
```

Set `RUN_DIR` to this directory and prepare the files on the toolchain server:

```bash
python3 samples/llm/gemma4/model/prepare.py \
  --source-dir "$RUN_DIR/artifacts/hpp/hbm" \
  --companion-dir "$RUN_DIR/artifacts/runtime_companions" \
  --output-dir /data/gemma4-e2b-sdk2
```

Transfer the prepared directory to S600 using your existing transfer method.
If its absolute path changes, run preparation with the new directory as both
`--source-dir` and `--output-dir` to verify and rebind the runtime configuration.
Alternatively, prepare separate language and companion directories directly on
the board. Preparation verifies full-file SHA256 and sizes, installs via a
temporary directory, and rejects existing mismatched files.

## Required SDK companion

SDK 2.0.0 GEMMA4 initialization requires
`gemma-4-E2B-it_vision_384x384_w8_nash-p_corenum_4.hbm` even for text requests.
It is supplied by the SDK's official
[Gemma4-E2B_w4.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E2B_w4.zip),
as listed in `model/resolve_model_s600.md`. Only this vision companion is used
from that bundle. The language model, embeddings and tokenizer remain bound to
our conversion. Its package entry and SHA256 are recorded independently;
vision accuracy is not claimed.

The eight files total 7,298,660,276 bytes. Preparation additionally writes
`oellm_config.json` and a manifest copy. The Archive package preserves these file identities; its download size differs from the extracted size.

Verify an existing directory without loading a runtime:

```bash
python3 samples/llm/gemma4/model/prepare.py --check --output-dir /data/gemma4-e2b-sdk2
```
