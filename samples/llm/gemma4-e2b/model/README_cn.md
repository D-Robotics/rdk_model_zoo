# 模型下载

**简体中文** | [English](./README.md)

Gemma4-E2B 预编译 HBM 模型（解压约 6 GB）托管在 HuggingFace：

```bash
export GEMMA4_HOME=~/gemma4_e2b   # 可选，默认值
bash download_model.sh
```

或手动下载：

```bash
pip install huggingface_hub
hf download ShockleyWong/gemma4-e2b-rdk-s100p --local-dir ~/gemma4_e2b
```

## 文件清单

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `model/gemma4-e2b_vit_ptq.hbm` | 329 MB | Vision 编码器 HBM |
| `model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm` | 4.5 GB | Text LLM HBM |
| `model/tok_embeddings.bin` | 1.5 GB | 外挂 token embedding 表 |
| `tokenizer/` | ~32 MB | tokenizer.json、chat template、config |

## 完整性校验（可选）

```bash
sha256sum ~/gemma4_e2b/model/*.hbm
# Vision: 470791849d21cffadb388cc61c8f4b1452078c1722d302fd8a8ac775ee9769f1
# Text:   3e4d4940051e4e8dc0cb434e972e7aae75d49504da3fac435e303f68af73a25f
```
