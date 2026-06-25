# Model Download

[简体中文](./README_cn.md) | **English**

Pre-compiled Gemma4-E2B HBM models (~6 GB unpacked) are hosted on HuggingFace:

```bash
export GEMMA4_HOME=~/gemma4_e2b   # optional, this is the default
bash download_model.sh
```

Or manually:

```bash
pip install huggingface_hub
hf download ShockleyWong/gemma4-e2b-rdk-s100p --local-dir ~/gemma4_e2b
```

## Files

| File | Size | Description |
| --- | --- | --- |
| `model/gemma4-e2b_vit_ptq.hbm` | 329 MB | Vision encoder HBM |
| `model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm` | 4.5 GB | Text LLM HBM |
| `model/tok_embeddings.bin` | 1.5 GB | External token embedding table |
| `tokenizer/` | ~32 MB | `tokenizer.json`, chat template, config |

## Integrity Check (optional)

```bash
sha256sum ~/gemma4_e2b/model/*.hbm
# Vision: 470791849d21cffadb388cc61c8f4b1452078c1722d302fd8a8ac775ee9769f1
# Text:   3e4d4940051e4e8dc0cb434e972e7aae75d49504da3fac435e303f68af73a25f
```
