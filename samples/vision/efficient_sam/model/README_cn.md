[English](README.md) | 简体中文

# EfficientSAM 模型准备

<a id="artifacts"></a>
## 制品清单

| Target | Encoder | Decoder | 格式 |
|---|---|---|---|
| x5 | `efficient_sam_vitt_encoder_512x512_default_none.bin` | `efficient_sam_vitt_decoder_fixedprompt_512_default.bin` | BIN |
| s100 | `nash-e/efficient_sam_vitt_encoder_512x512_nashe.hbm` | `nash-e/efficient_sam_vitt_decoder_512_nashe.hbm` | HBM |
| s100p | `nash-m/efficient_sam_vitt_encoder_512x512_nashm.hbm` | `nash-m/efficient_sam_vitt_decoder_512_nashm.hbm` | HBM |
| s600 | `nash-p/efficient_sam_vitt_encoder_512x512_nashp.hbm` | `nash-p/efficient_sam_vitt_decoder_512_nashp.hbm` | HBM |

<a id="preparation"></a>
## 准备步骤

在仓库根目录使用板端 Python 环境 显式执行目标命令。它会把两个 manifest 制品放到 `samples/vision/efficient_sam/model/`（X5）或其 `nash-e/`、`nash-m/`、`nash-p/` 子目录（S） 并报告观测摘要；需要网络，推理期间不会执行。

```bash
python3 samples/vision/efficient_sam/model/download.py --target s100
# 预期：samples/vision/efficient_sam/model/nash-e/ 下有两个文件
```

确切 URL 和文件身份来自 [X5 清单](../../../../docs/release/x5/models.yaml)、[S 清单](../../../../docs/release/s/models.yaml)。下载失败时可从另一台机器传入同一对文件，放到上述对应路径；runtime 不会替换成其他板型的配对。已有文件会核验而不会被覆盖。观测摘要只标识本地字节；发布方 SHA 未知时，它不能独立证明来源。自定义路径所需的确切 stage 制品 ID 可由本样例 runtime 的 `--list-models` 获得。

<a id="accompanying-files"></a>
## 伴随文件

`../test_data/dogs.jpg` 是随附输入 fixture，`../test_data/efficient_sam_binary_mask.png` 是保留的 source 参考 mask；二者都不是模型制品。

<a id="local-paths"></a>
## 本地路径

runtime 根据精确 manifest filename 在 `samples/vision/efficient_sam/model/` 下解析模型对。自定义 encoder 或 decoder 路径必须分别配对精确的 `--encoder-asset-id` 或 `--decoder-asset-id`。

<a id="formats-checksums"></a>
## 格式与校验值

x5 manifest 制品格式为 `bin`，S target 制品格式为 `hbm`。当前 manifest 每项均为 `sha256: null (unknown)`；不猜测或复制 digest。
