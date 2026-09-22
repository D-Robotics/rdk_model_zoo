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

<a id="preparation"></a>
## 准备

在仓库根目录执行：

```bash
# 输入：目标/变体对应的 Manifest 行 — 输出：本目录下的文件
# 成功判据：退出码 0，打印实测摘要；不残留半成品文件
bash samples/vision/efficientnet/model/download.sh x5 b2
bash samples/vision/efficientnet/model/download.sh s100 lite4
bash samples/vision/efficientnet/model/download.sh s600 lite0
```

Python 形式等价：
`python3 samples/vision/efficientnet/model/download.py --target s100 --variant lite1`。
省略变体时按 target 取源默认（x5 `b2`，s100/s600 `lite0`），与运行时
省略 `--variant` 的行为一致；显式 `--variant` 恒为精确选择。
下载器经同目录临时文件写入，核对内容长度与 Manifest 记录的发布方
SHA-256（如有），原子落盘且不覆盖已有文件（校验失败保留文件供排查）。
当前各行没有发布方 SHA-256，下载器打印实测摘要作为本地证据并声明来源
未经独立证明。下载是显式动作，绝不发生在推理过程中。

<a id="accompanying-files"></a>
## 随伴文件

分类运行还需要每行一个类别的 ImageNet 标签文件：
`datasets/imagenet/imagenet_classes.names`（X5 与 S 共用）。该文件已检入
仓库，无需下载。`test_data/` 内的标签副本是源分支遗留；规范路径是根
`datasets/` 下的那份。

<a id="local-paths"></a>
## 本地路径

准备完成后，制品位于 sample 的 `model/` 目录下（X5 平铺；S 系列在
`model/s100/` 与 `model/s600/` 子目录），相对 sample 根；[runtime/python/README_cn.md](../runtime/python/README_cn.md)
中的 `--model-path` 示例指向这些位置。源 S 交付把文件安装在
`/opt/hobot/model/<soc>/basic/` 下；该系统路径不再由本 sample 管理——如果
模型仍放在那里，请显式传入 `--model-path`。

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

Manifest 未记录这些行的发布方 SHA-256；未知值保持 `null（未知）`，
绝不在制品间复制。下载器每次下载都打印实测摘要作为本地证据。
