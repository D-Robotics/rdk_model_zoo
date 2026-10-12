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

<a id="directory"></a>
## 目录结构

```text
model/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── download.py  # 准备模型文件
├── download.sh  # 模型准备命令
└── download_model.sh  # Shell 脚本
```

<a id="preparation"></a>
## 准备步骤

在仓库根目录在主机或板卡上使用 Python 和 PyYAML显式执行目标命令。两个清单制品会放在 `samples/vision/mobile_sam/model/`（X5）或其 `nash-e/`、`nash-m/`、`nash-p/` 子目录（S） 并报告观测摘要；此步骤需要网络，推理期间不会下载。

```bash
python3 samples/vision/mobile_sam/model/download.py --target x5
python3 samples/vision/mobile_sam/model/download.py --target s100
python3 samples/vision/mobile_sam/model/download.py --target s100p
python3 samples/vision/mobile_sam/model/download.py --target s600
# 仅执行所需板卡的命令；每条命令准备一个编码器/解码器模型对。
```

确切 URL 和文件身份来自 [X5 清单](../../../../docs/release/x5/models.yaml)、[S 清单](../../../../docs/release/s/models.yaml)。下载失败时可从另一台机器传入同一对文件，放到上述对应路径；运行时不会替换成其他板型的配对。已有文件会核验而不会被覆盖。观测摘要只标识本地字节；发布方 SHA 未知时，它不能独立证明来源。自定义路径所需的确切阶段制品 ID 可由本样例运行时的 `--list-models` 获得。

<a id="accompanying-files"></a>
## 伴随文件

`../test_data/dogs.jpg` 是输入示例输入，`../test_data/mobile_sam_binary_mask.png` 是参考掩码。默认框为缩放后坐标中的 `[185,120,380,445]`。

<a id="local-paths"></a>
## 本地路径

运行时根据精确选择在 `samples/vision/mobile_sam/model/` 下解析模型对。自定义路径必须分别提供精确的编码器和解码器 asset ID。

<a id="formats-checksums"></a>
## 格式与校验值

x5 行使用 `bin`，S 行使用 `hbm`。当前清单每行均为 `sha256: null (unknown)`，不虚构校验值。
