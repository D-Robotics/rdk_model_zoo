# 模型下载

**简体中文** | [English](./README.md)

<a id="artifacts"></a>
## 发布组合

S100P（`nash-m`）与 S600（`nash-p`）各有 Vision/Text HBM，S100（`nash-e`）没有默认公共 HBM。
目录结构和文件名相同不意味着制品可跨目标使用。当前下载脚本检查文件是否非空后复用，并不自动校验已有文件的目标或 SHA-256。

<a id="preparation"></a>
## 显式准备

从仓库根目录进入 sample 的 model 目录，显式选择目标；以下为 S600。S100P 改用 `GEMMA4_SOC=s100p`，并使用独立目录。

```bash
cd samples/llm/gemma4-e2b/model
export GEMMA4_HOME=~/gemma4_e2b_s600
GEMMA4_SOC=s600 bash download_model.sh --dry-run
# 准备好下载时显式执行：
GEMMA4_SOC=s600 bash download_model.sh
```

公开的 `rdk_s100` 模型归档包含已验证的 S100P（`nash-m`）HBM，`rdk_s600` 包含已验证的 S600（`nash-p`）HBM，脚本按显式 `GEMMA4_SOC` 选择，不根据执行下载的电脑身份推断。S100（`nash-e`）请先将匹配的两个 HBM 放到 `$GEMMA4_HOME/model`，或显式提供对应模型目录 URL：

```bash
GEMMA4_HOME=~/gemma4_e2b_s100 GEMMA4_SOC=s100 GEMMA4_MODEL_BASE_URL=https://your-server/path/to/s100/model bash download_model.sh
```

## 准备接口

| 设置 / 参数 | 含义 |
| --- | --- |
| `GEMMA4_SOC` | 必填：`s100`、`s100p` 或 `s600`；不探测板型，也不默认回退 S100P |
| `GEMMA4_HOME` | 数据根目录；默认 `~/gemma4_e2b`；不同目标应使用独立目录 |
| `--dry-run` | 打印复用/下载决策及 URL；不联网、不创建目录 |
| `--help` | 只打印用法 |
| `GEMMA4_MODEL_BASE_URL` | 覆盖目标 HBM 目录 URL；S100 缺少本地 HBM 时必须提供 |
| `GEMMA4_COMMON_MODEL_BASE_URL` | 覆盖共享 embedding 目录 URL |
| `GEMMA4_TOKENIZER_BASE_URL` | 覆盖 tokenizer 目录 URL |

使用 Bash 执行；实际下载需要 PATH 中有 `wget`，预览不需要 wget 或板端 SDK。
未知参数、未指定目标均报错。S100 未指定 HBM URL 时，两个 HBM 必须已经存在，预览也遵循此规则。
脚本不检查模型内容：已有非空文件仍需由使用者确认属于正确目标。
更换 URL 或制品版本之前先清理旧 `.part` 文件，因为 `wget -c` 会续传它们。
传输失败保留部分文件并返回非零状态，空下载不会替换最终文件。

<a id="accompanying-files"></a>
## 配套文件

Token embedding 表和 tokenizer 在三个目标平台间共用，缺失时仍从公共归档下载。

<a id="local-paths"></a>
## 本地路径

$GEMMA4_HOME 下的文件布局（未设置时默认 `~/gemma4_e2b`）：

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

<a id="formats-checksums"></a>
## 完整性校验（可选）

```bash
sha256sum "$GEMMA4_HOME"/model/*.hbm
# S100P Vision: 470791849d21cffadb388cc61c8f4b1452078c1722d302fd8a8ac775ee9769f1
# S100P Text:   3e4d4940051e4e8dc0cb434e972e7aae75d49504da3fac435e303f68af73a25f
# S600 Vision:  a5998ca829cff121aa5672567b20e7be9f527da5b0220962b0fe7467bf8ff7b7
# S600 Text:    aab1831b1ea2b86763d5457890d89c55b684e4ba4834c1e008c668813d1cf646
```

以上四个 HBM 哈希保留自 S 源发布的 README。
活动发布清单目前对这些文件记录 `sha256: null`；下载器会打印观测到的摘要，使用者可将其与上面的发布摘要比较。HBM 是板端模型，
`tok_embeddings.bin` 是配套 embedding 数据，不是 X5 推理 BIN。下载使用 `.part` 临时文件并在完成后改名；
已有非空文件会跳过下载。`GEMMA4_MODEL_BASE_URL`、`GEMMA4_COMMON_MODEL_BASE_URL`、`GEMMA4_TOKENIZER_BASE_URL`
可显式替换三个来源。模型转换/量化流程见[转换指南](../conversion/README.md)。
