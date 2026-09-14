# Model Download

[简体中文](./README_cn.md) | **English**

The runtime uses the same filenames and directory layout on S100, S100P, and
S600, but the two HBM files must match the board SoC. The download script
detects the SoC and never substitutes an HBM compiled for another target.

```bash
export GEMMA4_HOME=~/gemma4_e2b   # optional, this is the default
bash download_model.sh
```

The public `rdk_s100` archive contains the validated S100P (`nash-m`)
HBMs, while `rdk_s600` contains the validated S600 (`nash-p`) HBMs. Both
are selected automatically. For S100 (`nash-e`), pre-place matching HBMs
under `$GEMMA4_HOME/model` or provide their directory URL explicitly:

```bash
GEMMA4_SOC=s100 GEMMA4_MODEL_BASE_URL=https://your-server/path/to/s100/model bash download_model.sh
```

The token embedding table and tokenizer are shared across all targets
and are downloaded from the common archive when missing.

The script downloads these runtime files:

```bash
model/gemma4-e2b_vit_ptq.hbm
model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm
model/tok_embeddings.bin
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
```

## Files

| File | Size | Description |
| --- | --- | --- |
| `model/gemma4-e2b_vit_ptq.hbm` | 329–377 MB | Platform-specific Vision encoder HBM |
| `model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm` | 4.5 GB | Text LLM HBM |
| `model/tok_embeddings.bin` | 1.5 GB | External token embedding table |
| `tokenizer/` | ~32 MB | `tokenizer.json`, chat template, config |

## Integrity Check (optional)

```bash
sha256sum ~/gemma4_e2b/model/*.hbm
# S100P Vision: 470791849d21cffadb388cc61c8f4b1452078c1722d302fd8a8ac775ee9769f1
# S100P Text:   3e4d4940051e4e8dc0cb434e972e7aae75d49504da3fac435e303f68af73a25f
# S600 Vision:  a5998ca829cff121aa5672567b20e7be9f527da5b0220962b0fe7467bf8ff7b7
# S600 Text:    aab1831b1ea2b86763d5457890d89c55b684e4ba4834c1e008c668813d1cf646
```
