# SDK 2.0 conversion

[简体中文](README_cn.md) · [Sample](../README.md)

Use the existing SDK container. The reference source is
`google/gemma-4-E2B-it-qat-q4_0-unquantized` at
`6befbaca7398925921802abd1f277b495b78b738`.
`main.py` reads SDK `released_models/gemma4_e2b/lm_w4_kv_w8` and `configs/gemma4.yml`,
binding local paths, the language branch and S600 settings. Quantization and
compilation remain native SDK operations.

```bash
python3 samples/llm/gemma4/conversion/main.py prepare \
  --sdk-root /opt/D-Robotics_LLM_2.0.0_S600_S100_SDK \
  --model-path /data/gemma-4-E2B-it-qat-q4_0-unquantized \
  --wikitext-dir /data/wikitext-2-raw-v1 --mmstar /data/MMStar.tsv \
  --output /data/gemma4-conversion
```

Run stages in this order: `autoclip`, `gptq`, `float`, `calib`, `eval`, `compile`:

```bash
python3 samples/llm/gemma4/conversion/main.py run \
  --work-dir /data/gemma4-conversion --stage autoclip
```

Repeat with the next stage name. Each call invokes only that native SDK stage
and preserves its exit code. Preparation requires a fresh output directory.
LightCompress calibrates with 128×512 WikiText2 train samples; HPP uses the
Gemma4 recipe's 30 MMStar samples. Evaluation uses the full WikiText2 test split
at sequence length 2048. Only the language branch is compiled.

Output is under `artifacts/hpp/hbm/`. Rebuilt artifacts require their own metadata,
size and SHA256 review before updating a model binding.

Reference environment: Python 3.10, PyTorch 2.8.0+cu128, HPP
3.3.5+cu128.torch280.qwen35, HBDK 4.14.1a2.dev202610090350+05dd1f0.develop.
Target: nash-p, four cores, chunk 512, cache 4096, external FP16 embeddings.
