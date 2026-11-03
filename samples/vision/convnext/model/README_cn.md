[English](README.md) | 简体中文

# ConvNeXt 模型制品

model 目录不检入任何二进制；制品由规范化下载器按平台发布 Manifest
显式获取。

<a id="artifacts"></a>
## 制品

| 文件 | Target | 变体 | 阶段 | 来源 |
| --- | --- | --- | --- | --- |
| `ConvNeXt_atto_224x224_nv12.bin` | x5 | atto | single | download（`x5:convnext:ConvNeXt_atto_224x224_nv12.bin`） |
| `ConvNeXt_nano_224x224_nv12.bin` | x5 | nano | single | download（`x5:convnext:ConvNeXt_nano_224x224_nv12.bin`） |
| `ConvNeXt_pico_224x224_nv12.bin` | x5 | pico | single | download（`x5:convnext:ConvNeXt_pico_224x224_nv12.bin`） |
| `ConvNeXt_femto_224x224_nv12.bin` | x5 | femto | single | download（`x5:convnext:ConvNeXt_femto_224x224_nv12.bin`） |

该引用是 `docs/release/x5/models.yaml` 中的一行；Manifest 是 URL 与格式
的事实源。X5 消费单个 packed NV12 张量，因此运行时总是把 `--model-path`
与精确引用配对使用。本模型没有 S 系列制品：S100/S100P/S600 的选择是显式
的 no-published-asset 错误，也不得在 S 板卡上复用 X5 文件。

<a id="directory"></a>
## 目录结构

```text
model/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── __init__.py  # Python 脚本
├── download.py  # 准备模型文件
└── download.sh  # 模型准备命令
```

<a id="preparation"></a>
## 准备

在仓库根目录执行：

```bash
# 输入：目标/变体对应的 Manifest 行 — 输出：本目录下的文件
# 输出：退出码 0，打印实测摘要；不残留半成品文件
bash samples/vision/convnext/model/download.sh x5
```

Python 形式等价：
`python3 samples/vision/convnext/model/download.py --target x5`。
下载器经同目录临时文件写入，核对内容长度与 Manifest 记录的发布方
SHA-256（如有），原子落盘且不覆盖已有文件（校验失败保留文件供排查）。
这些模型的发布方 SHA-256 未记录；下载器打印下载文件的 SHA-256。请先运行下载命令，再执行推理。

<a id="accompanying-files"></a>
## 随伴文件

分类运行还需要每行一个类别的 ImageNet 标签文件：
`datasets/imagenet/imagenet_classes.names`。该文件已检入仓库，无需下载。

<a id="local-paths"></a>
## 本地路径

准备完成后，制品平铺在 sample 的 `model/` 目录下，相对 sample 根；
[runtime/python/README_cn.md](../runtime/python/README_cn.md) 中的
`--model-path` 示例指向该位置。

<a id="formats-checksums"></a>
## 格式与校验值

| 文件 | 格式 | SHA-256 |
| --- | --- | --- |
| `ConvNeXt_atto_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入（224x224），F32 `[1,1000,1,1]` logits 输出 | null（未知） |
| `ConvNeXt_nano_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入（224x224），F32 `[1,1000,1,1]` logits 输出 | null（未知） |
| `ConvNeXt_pico_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入（224x224），F32 `[1,1000,1,1]` logits 输出 | null（未知） |
| `ConvNeXt_femto_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入（224x224），F32 `[1,1000,1,1]` logits 输出 | null（未知） |

Manifest 中的 SHA-256 为 `null (unknown)`。下载器打印下载文件的 SHA-256，供本地记录。

## Atto D2 已转换产物（2026-11-02）

`atto-d2` 是单独转换的 [timm/convnext_atto.d2_in1k](https://huggingface.co/timm/convnext_atto.d2_in1k/tree/435c0a8d02d5b872e90c8cde36887465982cb399)（Apache-2.0），权重版本固定为 `435c0a8d02d5b872e90c8cde36887465982cb399`。默认 `atto` 及四个历史产物的身份和历史数据保持原样。

新 BIN 为 4,528,264 字节，SHA-256 为 `c334e6c87ae202a0ef8d78b4506ecdab7fafb4684ce720fc4407f9bf990ea101`。导出和主机 PyTorch/ONNX 对照已通过；板端冒烟、精度、Runtime 1/2 并发、C++ 1/2 流均为 **not-run**。历史性能表不适用于此文件。预处理为 Pillow bicubic 短边缩放至 256、四舍五入居中裁剪 224，然后通过 OpenCV 转 BT.601 video-range NV12；ImageNet 归一化融合于 Mapper。输出为 1,000 类 float32 logits，Runtime 再做 softmax。此规格需要 Pillow。

```bash
# 仓库根目录：下载并验证已登记模型。
python3 samples/vision/convnext/model/download.py --target x5 --variant atto-d2
# 主机检查契约，不加载 SDK、不下载、不推理。
python3 samples/vision/convnext/runtime/python/main.py --target x5 --variant atto-d2 --dry-run
# 已准备好的 X5 板端命令；本次仅转换交付未执行。
python3 samples/vision/convnext/runtime/python/main.py \
  --target x5 --asset-id x5:convnext:convnext_atto_d2_bayese_224x224_nv12.bin \
  --model-path samples/vision/convnext/model/convnext_atto_d2_bayese_224x224_nv12.bin \
  --test-img samples/vision/convnext/test_data/cheetah.JPEG
```

<!-- converted-batch-20261102 -->
## 已转换新增规格（2026-11-02）

以下 timm 权重单独转换，已通过严格权重加载、结构重参数化及两组输入的
PyTorch/ONNX 主机对照、checker 和编译。板端冒烟、精度、Runtime 1/2 并发和
C++ 1/2 流均为 **not-run**，历史指标和截图不适用于这些产物。原有默认规格保留。
新规格需要 Pillow，使用 bicubic 短边 256、四舍五入居中裁剪 224；Mapper 融合
该权重实际的归一化参数。输出为 float32 1,000 类 logits，由原生 Runtime 做 softmax。

| 规格 | 模型文件 | 权重版本 | 板端 |
|---|---|---|---|
| `timm-femto-d1-in1k` | `convnext_femto_d1_in1k_bayese_224x224_nv12.bin` | [`1e0c02df687c`](https://huggingface.co/timm/convnext_femto.d1_in1k/blob/1e0c02df687c47abf0819e1a4f858293e17e0c50/README.md) | not-run |
| `timm-pico-d1-in1k` | `convnext_pico_d1_in1k_bayese_224x224_nv12.bin` | [`0d879cb0cfd9`](https://huggingface.co/timm/convnext_pico.d1_in1k/blob/0d879cb0cfd9b851a13f007d3ed170dd049f1476/README.md) | not-run |
| `timm-nano-d1h-in1k` | `convnext_nano_d1h_in1k_bayese_224x224_nv12.bin` | [`fe9cba6df2f1`](https://huggingface.co/timm/convnext_nano.d1h_in1k/blob/fe9cba6df2f17352639e23ca89e1aa6b37f51462/README.md) | not-run |

```bash
python3 samples/vision/convnext/model/download.py --target x5 --variant timm-femto-d1-in1k
python3 samples/vision/convnext/runtime/python/main.py --target x5 --variant timm-femto-d1-in1k --dry-run
```

精确 URL、SHA-256 和大小登记在 `docs/release/x5/models.yaml`；来源及主机对照
记录见 `conversion/converted.json`。模型许可逐项保留在公开 release 清单中。
<!-- /converted-batch-20261102 -->

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

这些独立规格已完成转换、完整 OE 和实际 HBM 接口核验。源权重、ONNX、每目标配置、校准数量和摘要见 [converted_targets_20261103.json](../conversion/converted_targets_20261103.json)。

| 目标 | Native variant | 组件 | 校准图片 |
| --- | --- | --- | ---: |
| s100 | `timm-femto-d1-in1k-s-int16-o1-20261103` | model | 200 |

```bash
python3 samples/vision/convnext/model/download.py --target s100 --variant timm-femto-d1-in1k-s-int16-o1-20261103
python3 samples/vision/convnext/runtime/python/main.py --target s100 --variant timm-femto-d1-in1k-s-int16-o1-20261103 --asset-id s:convnext:s100/convnext_femto_d1_in1k_nashe_224x224_nv12_float_int16_20261103_o1.hbm --model-path /path/to/convnext_femto_d1_in1k_nashe_224x224_nv12_float_int16_20261103_o1.hbm --test-img /path/to/image.jpg
```

命令中的路径需要替换为实际文件。每套多组件模型须同时准备全部组件。旧默认与既有指标保留原身份；本轮规格的板端 smoke、数据集精度、Runtime 1/2 并发和 C++ 1/2 流均为 `not-run`。
