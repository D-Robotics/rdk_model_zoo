# 模型下载

**简体中文** | [English](./README.md)

S100、S100P 与 S600 使用相同的文件名和目录结构，但两个 HBM 文件必须与板端 SoC 匹配。下载脚本会自动识别 SoC，不会用其他平台 HBM 替代当前平台模型。

```bash
export GEMMA4_HOME=~/gemma4_e2b   # 可选，默认值
bash download_model.sh
```

公开的 `rdk_s100` 模型归档包含已验证的 S100P（`nash-m`）HBM，`rdk_s600` 包含已验证的 S600（`nash-p`）HBM，两者都会按 SoC 自动选择。S100（`nash-e`）请先将匹配的两个 HBM 放到 `$GEMMA4_HOME/model`，或显式提供对应模型目录 URL：

```bash
GEMMA4_SOC=s100 GEMMA4_MODEL_BASE_URL=https://your-server/path/to/s100/model bash download_model.sh
```

Token embedding 表和 tokenizer 在三个目标平台间共用，缺失时仍从公共归档下载。

脚本会下载以下运行时文件：

```bash
model/gemma4-e2b_vit_ptq.hbm
model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm
model/tok_embeddings.bin
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
```

## 文件清单

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `model/gemma4-e2b_vit_ptq.hbm` | 329–377 MB | 平台对应的 Vision 编码器 HBM |
| `model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm` | 4.5 GB | Text LLM HBM |
| `model/tok_embeddings.bin` | 1.5 GB | 外挂 token embedding 表 |
| `tokenizer/` | ~32 MB | tokenizer.json、chat template、config |

## 完整性校验（可选）

```bash
sha256sum ~/gemma4_e2b/model/*.hbm
# S100P Vision: 470791849d21cffadb388cc61c8f4b1452078c1722d302fd8a8ac775ee9769f1
# S100P Text:   3e4d4940051e4e8dc0cb434e972e7aae75d49504da3fac435e303f68af73a25f
# S600 Vision:  a5998ca829cff121aa5672567b20e7be9f527da5b0220962b0fe7467bf8ff7b7
# S600 Text:    aab1831b1ea2b86763d5457890d89c55b684e4ba4834c1e008c668813d1cf646
```
