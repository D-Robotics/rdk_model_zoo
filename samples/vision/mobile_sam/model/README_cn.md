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

x5 行使用 `bin`，S 行使用 `hbm`。历史发布行中的 SHA 仍为 `null (unknown)`；下方带日期的转换配置记录了核验后的 SHA-256。

## 本轮 S100 转换规格（2026-11-03）

`tinyvit-512-int16-20261103-o1` · S100 / nash-e · OE 3.7.0 (hb_compile 3.5.3, HMCT 2.6.5, HBDK 4.7.5) · O1.

显式选择本轮独立规格。原默认规格及其历史数据保留。实际 HBM 输出为 float32。

| Role | Exact asset ID | SHA-256 | Bytes |
| --- | --- | --- | ---: |
| encoder | `s:mobile_sam:nash-e/mobile_sam_image_encoder_norm_512x512_nashe_int16_20261103_o1.hbm` | `19f6f39b85c276bbd5f608bbfa3829a03a9507fe0b70a27963c01a03aa9bb8d6` | 49086032 |
| decoder | `s:mobile_sam:nash-e/mobile_sam_decoder_512_nashe_int16_20261103_o1.hbm` | `ea123331de46a21f92a022ba6c6730305ba75ba744efa405f33d08120397e6ad` | 14175872 |

```bash
python3 samples/vision/mobile_sam/model/download.py --target s100 --variant tinyvit-512-int16-20261103-o1
python3 samples/vision/mobile_sam/runtime/python/main.py --target s100 --variant tinyvit-512-int16-20261103-o1 --test-img samples/vision/mobile_sam/test_data/dogs.jpg
```

校准图片数量：198；输入来自独立的 COCO 训练图片。板端 smoke、精度、Runtime 1/2 并发和 C++ 1/2 流均为 `not-run`。源码、checkpoint、ONNX、实际配置、原生配方与完整 OE 的摘要见 [converted.json](../conversion/converted.json)。

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

这些独立规格已完成转换、完整 OE 和实际 HBM 接口核验。源权重、ONNX、每目标配置、校准数量和摘要见 [converted_targets_20261103.json](../conversion/converted_targets_20261103.json)。

| 目标 | Native variant | 组件 | 校准图片 |
| --- | --- | --- | ---: |
| s100p | `tinyvit-512-int16-20261103-o1` | encoder, decoder | 198 |

```bash
python3 samples/vision/mobile_sam/model/download.py --target s100p --variant tinyvit-512-int16-20261103-o1
python3 samples/vision/mobile_sam/runtime/python/main.py --target s100p --variant tinyvit-512-int16-20261103-o1 --encoder-model-path /path/to/mobile_sam_image_encoder_norm_512x512_nashm_int16_20261103_o1.hbm --encoder-asset-id s:mobile_sam:nash-m/mobile_sam_image_encoder_norm_512x512_nashm_int16_20261103_o1.hbm --decoder-model-path /path/to/mobile_sam_decoder_512_nashm_int16_20261103_o1.hbm --decoder-asset-id s:mobile_sam:nash-m/mobile_sam_decoder_512_nashm_int16_20261103_o1.hbm --test-img samples/vision/mobile_sam/test_data/dogs.jpg
```

命令中的路径需要替换为实际文件。每套多组件模型须同时准备全部组件。旧默认与既有指标保留原身份；本轮规格的板端 smoke、数据集精度、Runtime 1/2 并发和 C++ 1/2 流均为 `not-run`。
