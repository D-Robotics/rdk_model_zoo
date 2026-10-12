[English](README.md) | 简体中文

# EfficientNet 模型制品

model 目录不检入任何二进制；制品由规范化下载器按平台发布 Manifest
显式获取。

<a id="artifacts"></a>
## 制品

| 文件 | Target | 变体 | 阶段 | 来源 |
| --- | --- | --- | --- | --- |
| `EfficientNet_B2_224x224_nv12.bin` | x5 | b2 | single | download（`x5:efficientnet:EfficientNet_B2_224x224_nv12.bin`） |
| `EfficientNet_B3_224x224_nv12.bin` | x5 | b3 | single | download（`x5:efficientnet:EfficientNet_B3_224x224_nv12.bin`） |
| `EfficientNet_B4_224x224_nv12.bin` | x5 | b4 | single | download（`x5:efficientnet:EfficientNet_B4_224x224_nv12.bin`） |
| `s100/efficientnet_lite0_224x224_nv12.hbm` | s100 | lite0 | single | download（`s:efficientnet:s100/efficientnet_lite0_224x224_nv12.hbm`） |
| `s100/efficientnet_lite1_240x240_nv12.hbm` | s100 | lite1 | single | download（`s:efficientnet:s100/efficientnet_lite1_240x240_nv12.hbm`） |
| `s100/efficientnet_lite2_260x260_nv12.hbm` | s100 | lite2 | single | download（`s:efficientnet:s100/efficientnet_lite2_260x260_nv12.hbm`） |
| `s100/efficientnet_lite3_300x300_nv12.hbm` | s100 | lite3 | single | download（`s:efficientnet:s100/efficientnet_lite3_300x300_nv12.hbm`） |
| `s100/efficientnet_lite4_380x380_nv12.hbm` | s100 | lite4 | single | download（`s:efficientnet:s100/efficientnet_lite4_380x380_nv12.hbm`） |
| `s600/efficientnet_lite0_224x224_nv12.hbm` | s600 | lite0 | single | download（`s:efficientnet:s600/efficientnet_lite0_224x224_nv12.hbm`） |
| `s600/efficientnet_lite1_240x240_nv12.hbm` | s600 | lite1 | single | download（`s:efficientnet:s600/efficientnet_lite1_240x240_nv12.hbm`） |
| `s600/efficientnet_lite2_260x260_nv12.hbm` | s600 | lite2 | single | download（`s:efficientnet:s600/efficientnet_lite2_260x260_nv12.hbm`） |
| `s600/efficientnet_lite3_300x300_nv12.hbm` | s600 | lite3 | single | download（`s:efficientnet:s600/efficientnet_lite3_300x300_nv12.hbm`） |
| `s600/efficientnet_lite4_380x380_nv12.hbm` | s600 | lite4 | single | download（`s:efficientnet:s600/efficientnet_lite4_380x380_nv12.hbm`） |

每条引用都是 `docs/release/x5/models.yaml` 或 `docs/release/s/models.yaml`
中的一行；Manifest 是 URL 与格式的事实源。X5 消费单个 packed NV12 张量；
S100/S600 消费分离的 Y 与 UV 张量——裸文件名无法表达协议，因此运行时总是
把 `--model-path` 与精确引用配对使用。lite 系列具有**逐变体几何**
（224/240/260/300/380），由变体而非默认尺寸决定。S100P 没有资产行，
也不能用 S100 文件替代。

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
bash samples/vision/efficientnet/model/download.sh x5 b2
bash samples/vision/efficientnet/model/download.sh s100 lite4
bash samples/vision/efficientnet/model/download.sh s600 lite0
```

Python 形式等价：
`python3 samples/vision/efficientnet/model/download.py --target s100 --variant lite1`。
省略变体时按 target 取默认值（x5 `b2`，s100/s600 `lite0`），与运行时
省略 `--variant` 的行为一致；显式 `--variant` 恒为精确选择。
下载器经同目录临时文件写入，核对内容长度与 Manifest 记录的发布方
SHA-256（如有），原子落盘且不覆盖已有文件（校验失败保留文件供排查）。
这些模型的发布方 SHA-256 未记录；下载器打印下载文件的 SHA-256。请先运行下载命令，再执行推理。

<a id="accompanying-files"></a>
## 随伴文件

分类运行使用 ImageNet 标签文件
`datasets/imagenet/imagenet_classes.names`（X5 与 S 共用）；运行时
`--label-file` 的默认值即该路径，文件已检入仓库，无需下载。`test_data/`
内的 `imagenet_classes.names`、`imagenet1000_labels.txt` 与
`imagenet_1k.json` 是 ImageNet-1k 标签的其它格式，需要时经 `--label-file`
显式指定。

<a id="local-paths"></a>
## 本地路径

准备完成后，模型位于样例的 `model/` 目录下：X5 文件直接放在该目录，S 系列放在 `model/s100/` 或 `model/s600/`。[运行示例](../runtime/python/README_cn.md)中的 `--model-path` 使用这些路径。若使用 `/opt/hobot/model/<soc>/basic/` 中的模型，请通过 `--model-path` 指定文件。

<a id="formats-checksums"></a>
## 格式与校验值

| 文件 | 格式 | SHA-256 |
| --- | --- | --- |
| `EfficientNet_B2_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入（224x224），F32 `[1,1000,1,1]` logits 输出 | null（未知） |
| `EfficientNet_B3_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入（224x224），F32 `[1,1000,1,1]` logits 输出 | null（未知） |
| `EfficientNet_B4_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入（224x224），F32 `[1,1000,1,1]` logits 输出 | null（未知） |
| `s100/efficientnet_lite0_224x224_nv12.hbm` | nash-e `.hbm`，split Y/UV 输入（224x224），F32 `[1,1000]` logits 输出 | null（未知） |
| `s100/efficientnet_lite1_240x240_nv12.hbm` | nash-e `.hbm`，split Y/UV 输入（240x240），F32 `[1,1000]` logits 输出 | null（未知） |
| `s100/efficientnet_lite2_260x260_nv12.hbm` | nash-e `.hbm`，split Y/UV 输入（260x260），F32 `[1,1000]` logits 输出 | null（未知） |
| `s100/efficientnet_lite3_300x300_nv12.hbm` | nash-e `.hbm`，split Y/UV 输入（300x300），F32 `[1,1000]` logits 输出 | null（未知） |
| `s100/efficientnet_lite4_380x380_nv12.hbm` | nash-e `.hbm`，split Y/UV 输入（380x380），F32 `[1,1000]` logits 输出 | null（未知） |
| `s600/efficientnet_lite0_224x224_nv12.hbm` | nash-p `.hbm`，split Y/UV 输入（224x224），F32 `[1,1000]` logits 输出 | null（未知） |
| `s600/efficientnet_lite1_240x240_nv12.hbm` | nash-p `.hbm`，split Y/UV 输入（240x240），F32 `[1,1000]` logits 输出 | null（未知） |
| `s600/efficientnet_lite2_260x260_nv12.hbm` | nash-p `.hbm`，split Y/UV 输入（260x260），F32 `[1,1000]` logits 输出 | null（未知） |
| `s600/efficientnet_lite3_300x300_nv12.hbm` | nash-p `.hbm`，split Y/UV 输入（300x300），F32 `[1,1000]` logits 输出 | null（未知） |
| `s600/efficientnet_lite4_380x380_nv12.hbm` | nash-p `.hbm`，split Y/UV 输入（380x380），F32 `[1,1000]` logits 输出 | null（未知） |

Manifest 中的 SHA-256 为 `null (unknown)`。下载器打印下载文件的 SHA-256，供本地记录。
