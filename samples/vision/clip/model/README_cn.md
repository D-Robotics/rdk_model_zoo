[English](README.md) | 简体中文

# 模型制品 — CLIP 编码器对

<a id="artifacts"></a>
## 制品清单

示例需要两个独立制品：BPU 图像编码器和 CPU ONNX 文本编码器。分别提供图像编码器和文本编码器的 asset ID 与模型路径。

| 制品 | 格式 | Target | 阶段 | 来源 |
| --- | --- | --- | --- | --- |
| `img_encoder.bin` | BIN | x5 | image 编码器 | download |
| `text_encoder.onnx` | ONNX | x5 | text 编码器 | download |

清单精确 URL：

| Asset ID | 发布 URL |
| --- | --- |
| `x5:clip:img_encoder.bin` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/clip/img_encoder.bin> |
| `x5:clip:text_encoder.onnx` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/clip/text_encoder.onnx> |

<a id="directory"></a>
## 目录结构

```text
model/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── download.py  # 准备模型文件
└── download.sh  # 模型准备命令
```

<a id="preparation"></a>
## 准备步骤

从仓库根目录运行。目标板卡是命名参数且只接受 `x5`；没有 variant 参数。`runtime/python/run.sh` 不会隐式执行下载。

```bash
# cwd：仓库根目录；来源：上表精确 URL
python3 samples/vision/clip/model/download.py --target x5
# 预期：samples/vision/clip/model/img_encoder.bin 和 text_encoder.onnx

# cwd：仓库根目录；等价 shell wrapper 和显式落盘目录
bash samples/vision/clip/model/download.sh --target x5 --output-dir /tmp/clip-model
# 预期：/tmp/clip-model/img_encoder.bin 和 /tmp/clip-model/text_encoder.onnx
```

清单将两个 SHA-256 均记录为未知。下载器打印观测 digest，并说明无法对照已发布数值校验。I/O 或选择错误退出 2。

<a id="accompanying-files"></a>
## 伴随文件

| 文件 | 作用 | 必需 |
| --- | --- | --- |
| `download.py` | 下载两个精确清单制品。 | 脚本准备时是；两个文件都存在时否。 |
| `download.sh` | `download.py` 的命名参数脚本。 | 否。 |
| `../runtime/python/bpe_simple_vocab_16e6.txt.gz` | 文本 tokenizer 使用的源 BPE 词表。 | 文本编码时是。 |

<a id="local-paths"></a>
## 本地路径

- 默认图像路径：`samples/vision/clip/model/img_encoder.bin`。
- 默认文本路径：`samples/vision/clip/model/text_encoder.onnx`。
- 传入 `--image-model-path` 时必须配对 `--image-asset-id x5:clip:img_encoder.bin`。
- 传入 `--text-model-path` 时必须配对 `--text-asset-id x5:clip:text_encoder.onnx`。

<a id="formats-checksums"></a>
## 格式与校验值

| 制品 | 格式 | SHA-256 | 数值来源 |
| --- | --- | --- | --- |
| `img_encoder.bin` | BIN | `sha256: null (unknown)` | `docs/release/x5/models.yaml` |
| `text_encoder.onnx` | ONNX | `sha256: null (unknown)` | `docs/release/x5/models.yaml` |

## 许可

准备代码遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。模型制品沿用发布记录中的来源和许可证元数据；不在此新增断言。

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

这些独立规格已完成转换、完整 OE 和实际 HBM 接口核验。源权重、ONNX、每目标配置、校准数量和摘要见 [converted_targets_20261103.json](../conversion/converted_targets_20261103.json)。

| 目标 | Native variant | 组件 | 校准图片 |
| --- | --- | --- | ---: |
| s100 | `vit-b32-native-int16-o1-20261103` | image, text | 200 |
| s100p | `vit-b32-native-int16-o1-20261103` | image, text | 200 |
| s600 | `vit-b32-native-int16-o1-20261103` | image, text | 200 |

```bash
python3 samples/vision/clip/model/download.py --target s100 --variant vit-b32-native-int16-o1-20261103
python3 samples/vision/clip/runtime/python/main.py --target s100 --variant vit-b32-native-int16-o1-20261103 --image-model-path /path/to/clip_vitb32_image_nashe_int16_20261103_o1.hbm --image-asset-id s:clip:s100/clip_vitb32_image_nashe_int16_20261103_o1.hbm --text-model-path /path/to/clip_vitb32_text_20261103.onnx --text-asset-id s:clip:s100/clip_vitb32_text_20261103.onnx --test-img samples/vision/clip/test_data/dog.jpg --texts 'a diagram,a dog'
```

命令中的路径需要替换为实际文件。每套多组件模型须同时准备全部组件。旧默认与既有指标保留原身份；本轮规格的板端 smoke、数据集精度、Runtime 1/2 并发和 C++ 1/2 流均为 `not-run`。
