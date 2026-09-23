[English](README.md) | 简体中文

# MobileSAM 模型准备

<a id="artifacts"></a>
## 制品清单

| Target | Encoder | Decoder | 格式 |
|---|---|---|---|
| x5 | `mobile_sam_image_encoder_norm_512x512_allint16.bin` | `mobile_sam_decoder_512_box_default.bin` | BIN |
| s100 | `nash-e/mobile_sam_image_encoder_norm_512x512_nashe.hbm` | `nash-e/mobile_sam_decoder_512_nashe.hbm` | HBM |
| s100p | `nash-m/mobile_sam_image_encoder_norm_512x512_nashm.hbm` | `nash-m/mobile_sam_decoder_512_nashm.hbm` | HBM |
| s600 | `nash-p/mobile_sam_image_encoder_norm_512x512_nashp.hbm` | `nash-p/mobile_sam_decoder_512_nashp.hbm` | HBM |

<a id="preparation"></a>
## 准备步骤

在仓库根目录使用板端 Python 环境 显式执行目标命令。两个 manifest 制品会放在 `samples/vision/mobile_sam/model/`（X5）或其 `nash-e/`、`nash-m/`、`nash-p/` 子目录（S） 并报告观测摘要；此步骤需要网络，推理期间不会下载。

```bash
python3 samples/vision/mobile_sam/model/download.py --target s100
# 预期：samples/vision/mobile_sam/model/nash-e/ 下有两个文件
```

确切 URL 和文件身份来自 [X5 清单](../../../../docs/release/x5/models.yaml)、[S 清单](../../../../docs/release/s/models.yaml)。下载失败时可从另一台机器传入同一对文件，放到上述对应路径；runtime 不会替换成其他板型的配对。已有文件会核验而不会被覆盖。观测摘要只标识本地字节；发布方 SHA 未知时，它不能独立证明来源。自定义路径所需的确切 stage 制品 ID 可由本样例 runtime 的 `--list-models` 获得。

<a id="accompanying-files"></a>
## 伴随文件

`../test_data/dogs.jpg` 是输入 fixture，`../test_data/mobile_sam_binary_mask.png` 是保留的 source 参考 mask。默认框为 resize 后坐标中的 `[185,120,380,445]`。

<a id="local-paths"></a>
## 本地路径

runtime 根据精确选择在 `samples/vision/mobile_sam/model/` 下解析模型对。自定义路径必须分别提供精确的 encoder 和 decoder asset ID。

<a id="formats-checksums"></a>
## 格式与校验值

x5 行使用 `bin`，S 行使用 `hbm`。当前 manifest 每行均为 `sha256: null (unknown)`，不虚构校验值。
