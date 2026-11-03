[English](README.md) | 简体中文

# MODNet 模型

<a id="artifacts"></a>
## 制品清单

| 目标板卡 | 阶段 | asset ID | 文件 | 获取方式 |
|---|---|---|---|---|
| X5 | matting | `x5:modnet:modnet_512x512_rgb.bin` | `modnet_512x512_rgb.bin` | 外部手工制品 |

摄影人像权重另有独立的 S100/S100P/S600 HBM 制品，精确 URL 与摘要见下方转换规格；板端验收为 `not-run`。

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

历史手工制品没有公开 URL。通过授权模型提供方取得外部模型，确认身份为 `x5:modnet:modnet_512x512_rgb.bin`，并放到 `samples/vision/modnet/model/modnet_512x512_rgb.bin`。下方准备命令会提示模型放置位置：

```bash
# cwd：仓库根目录
python3 -m samples.vision.modnet.model.download \
  --target x5 --asset-id x5:modnet:modnet_512x512_rgb.bin \
  --output-dir samples/vision/modnet/model
```

它会打印手工准备要求并返回 `2`；由于没有 URL，这是预期行为。

### 公开摄影人像制品

新制品使用 `x5:modnet:modnet_photographic_portrait_512x512_rgb_bayese_20261102.bin`，文件为 `modnet_photographic_portrait_512x512_rgb_bayese_20261102.bin`。完成清单登记后按以下命令获取；URL 和 SHA-256 以平台清单为准，下载器会校验已有文件，不覆盖旧模型。

```bash
python3 -m samples.vision.modnet.model.download --target x5 \
  --asset-id x5:modnet:modnet_photographic_portrait_512x512_rgb_bayese_20261102.bin
```

公开权重与模型采用上游 Apache-2.0；精确源码、权重摘要和预处理见[转换说明](../conversion/README_cn.md)。板端性能和抠像精度待补测。

<a id="accompanying-files"></a>
## 伴随文件

- `../test_data/person.jpg`：源输入图像。
- `../test_data/bg.jpg`：可选背景合成图像。

<a id="local-paths"></a>
## 本地路径

运行时默认路径为 `samples/vision/modnet/model/modnet_512x512_rgb.bin`。外部路径必须配合精确 `--asset-id x5:modnet:modnet_512x512_rgb.bin`，由该 ID 绑定协议身份。

<a id="formats-checksums"></a>
## 格式与校验值

期望制品是 X5 `bin` 部署模型。旧条目记录 `url: null` 和 `sha256: null (unknown)`。任何本地观测都不视为发布者认证。

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

这些独立规格已完成转换、完整 OE 和实际 HBM 接口核验。源权重、ONNX、每目标配置、校准数量和摘要见 [converted_targets_20261103.json](../conversion/converted_targets_20261103.json)。

| 目标 | Native variant | 组件 | 校准图片 |
| --- | --- | --- | ---: |
| s100 | `photographic-portrait-512-s-o1-20261103` | model | 200 |
| s100p | `photographic-portrait-512-s-o1-20261103` | model | 200 |
| s600 | `photographic-portrait-512-s-o1-20261103` | model | 200 |

```bash
python3 samples/vision/modnet/model/download.py --target s100 --asset-id s:modnet:s100/modnet_photographic_portrait_512x512_rgb_nashe_20261103_o1.hbm
python3 samples/vision/modnet/runtime/python/main.py --target s100 --asset-id s:modnet:s100/modnet_photographic_portrait_512x512_rgb_nashe_20261103_o1.hbm --model-path /path/to/modnet_photographic_portrait_512x512_rgb_nashe_20261103_o1.hbm --test-img samples/vision/modnet/test_data/person.jpg
```

命令中的路径需要替换为实际文件。每套多组件模型须同时准备全部组件。旧默认与既有指标保留原身份；本轮规格的板端 smoke、数据集精度、Runtime 1/2 并发和 C++ 1/2 流均为 `not-run`。
