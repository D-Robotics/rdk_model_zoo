[English](README.md) | 简体中文

# MobileNetV4 模型制品

model 目录不提交任何二进制；制品由 下载器按平台发布 Manifest
显式获取。

<a id="artifacts"></a>
## 制品清单

| 文件 | Target | 变体 | 阶段 | 来源 |
| --- | --- | --- | --- | --- |
| `mobilenetv4_conv_small_bayese_224x224_nv12.bin` | x5 | small | 单阶段 | 下载（`x5:mobilenetv4:mobilenetv4_conv_small_bayese_224x224_nv12.bin`） |
| `s100/mobilenetv4_conv_small_nashe_224x224_nv12.hbm` | s100 | small | 单阶段 | 下载（`s:mobilenetv4:s100/mobilenetv4_conv_small_nashe_224x224_nv12.hbm`） |
| `s100p/mobilenetv4_conv_small_nashm_224x224_nv12.hbm` | s100p | small | 单阶段 | 下载（`s:mobilenetv4:s100p/mobilenetv4_conv_small_nashm_224x224_nv12.hbm`） |
| `s600/mobilenetv4_conv_small_nashp_224x224_nv12.hbm` | s600 | small | 单阶段 | 下载（`s:mobilenetv4:s600/mobilenetv4_conv_small_nashp_224x224_nv12.hbm`） |
| `mobilenetv4_conv_medium_bayese_224x224_nv12.bin` | x5 | medium | 单阶段 | 下载（`x5:mobilenetv4:mobilenetv4_conv_medium_bayese_224x224_nv12.bin`） |
| `s100/mobilenetv4_conv_medium_nashe_224x224_nv12.hbm` | s100 | medium | 单阶段 | 下载（`s:mobilenetv4:s100/mobilenetv4_conv_medium_nashe_224x224_nv12.hbm`） |
| `s100p/mobilenetv4_conv_medium_nashm_224x224_nv12.hbm` | s100p | medium | 单阶段 | 下载（`s:mobilenetv4:s100p/mobilenetv4_conv_medium_nashm_224x224_nv12.hbm`） |
| `s600/mobilenetv4_conv_medium_nashp_224x224_nv12.hbm` | s600 | medium | 单阶段 | 下载（`s:mobilenetv4:s600/mobilenetv4_conv_medium_nashp_224x224_nv12.hbm`） |

每个引用对应 `docs/release/x5/models.yaml` 或 `docs/release/s/models.yaml`
中的一行；URL 与格式以 Manifest 为权威。X5 消费一个 packed NV12 张量；
S100/S100P/S600 消费分离的 Y 与 UV 张量——仅凭文件名无法确定协议，因此
runtime 始终将 `--model-path` 与完整引用配对。每个平台使用与其工具链
march 对应的专用制品，S100P 不能使用 S100 的文件。

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
## 准备步骤

在仓库根目录执行：

```bash
# 输入：目标/变体对应的 Manifest 行 — 输出：本目录下的制品文件
# 成功判据：退出码 0，打印观察 digest；不留半成品文件
bash samples/vision/mobilenetv4/model/download.sh x5 small
bash samples/vision/mobilenetv4/model/download.sh x5 medium
bash samples/vision/mobilenetv4/model/download.sh s100p medium
```

Python 形式等价：
`python3 samples/vision/mobilenetv4/model/download.py --target s100 --variant small`。下载器先写同目录临时文件，按 Manifest 中的
SHA-256 校验，原子落盘且不覆盖已有文件（校验失败时保留原文件供调查）。
下载后打印的 digest 应与下表中的 SHA-256 一致。下载是显式操作，推理过程不会
触发下载。

<a id="accompanying-files"></a>
## 伴随文件

分类器运行使用 ImageNet 标签文件
`datasets/imagenet/imagenet_classes.names`（X5 与 S 系列共用）；运行时
`--label-file` 的默认值即该路径，文件在仓库内，无需下载。`test_data/`
内的 `imagenet_classes.names` 内容与该文件相同，需要时经 `--label-file`
显式指定。

<a id="local-paths"></a>
## 本地路径

准备完成后，制品位于 sample 的 `model/` 目录（X5 平铺，S 系列在
`model/s100/`、`model/s100p/`、`model/s600/` 子目录），相对 sample 根目录；
[runtime/python/README_cn.md](../runtime/python/README_cn.md) 中的
`--model-path` 示例即指向这些位置。

<a id="formats-checksums"></a>
## 格式与校验值

| 文件 | 格式 | SHA-256 |
| --- | --- | --- |
| `mobilenetv4_conv_small_bayese_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入 (224x224), F32 `[1,1000,1,1]` logits 输出 | `3978f150ce77e725c80c1624f39300a56b8def69543449f6dd3fc9da9a46ff1b` |
| `s100/mobilenetv4_conv_small_nashe_224x224_nv12.hbm` | nash-e `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `ae8730fab716b4a7046304e10c855bc9db5c8d2385a569e349b315f8f98446d5` |
| `s100p/mobilenetv4_conv_small_nashm_224x224_nv12.hbm` | nash-m `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `34114e3f00030a9a01aea4f9634387be6441b57603267f59ef9f9d9190d5ec10` |
| `s600/mobilenetv4_conv_small_nashp_224x224_nv12.hbm` | nash-p `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `a2bb74d2051c8495827d14071676a7320cbb2637507b424822388608269298a2` |
| `mobilenetv4_conv_medium_bayese_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入 (224x224), F32 `[1,1000,1,1]` logits 输出 | `0a1dc43e2a6d9b789112307f2f2c681d3abfd1406aaef3fde3c3a1d3870f9025` |
| `s100/mobilenetv4_conv_medium_nashe_224x224_nv12.hbm` | nash-e `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `adec594b442eb5efcd2ff1694b86f087d630e2a933871bffaeebea7ae956ffca` |
| `s100p/mobilenetv4_conv_medium_nashm_224x224_nv12.hbm` | nash-m `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `b95804313b774a49c3ebc48a3f3ca278964c93df456822ac47b9ee6abc133107` |
| `s600/mobilenetv4_conv_medium_nashp_224x224_nv12.hbm` | nash-p `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `5b1b0abc231e6d0272a39de32885ef46eb3755e808763def5efbccaaa73d415b` |

所有制品都是对固定版本 [timm](https://github.com/huggingface/pytorch-image-models)
检查点（`mobilenetv4_conv_small.e2400_r224_in1k` 与
`mobilenetv4_conv_medium.e500_r224_in1k`，Apache-2.0）做训练后 INT8 量化的结果，
并按文件名中的 march 标记编译：`bayese`（bayes-e，X5）、`nashe`（nash-e，
S100）、`nashm`（nash-m，S100P）、`nashp`（nash-p，S600）。网络输入的归一化
已编译进模型；runtime 送入的是由 224x224 中心裁剪图转换的 NV12
（见[转换说明](../conversion/README_cn.md#preprocessing)）。
