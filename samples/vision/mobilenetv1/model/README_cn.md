[English](README.md) | 简体中文

# MobileNetV1 模型制品

model 目录不提交任何二进制；制品由 下载器按平台发布 Manifest
显式获取。

<a id="artifacts"></a>
## 制品清单

| 文件 | Target | 变体 | 阶段 | 来源 |
| --- | --- | --- | --- | --- |
| `mobilenetv1_100_bayese_224x224_nv12.bin` | x5 | 100 | 单阶段 | 下载（`x5:mobilenetv1:mobilenetv1_100_bayese_224x224_nv12.bin`） |
| `s100/mobilenetv1_100_nashe_224x224_nv12.hbm` | s100 | 100 | 单阶段 | 下载（`s:mobilenetv1:s100/mobilenetv1_100_nashe_224x224_nv12.hbm`） |
| `s100p/mobilenetv1_100_nashm_224x224_nv12.hbm` | s100p | 100 | 单阶段 | 下载（`s:mobilenetv1:s100p/mobilenetv1_100_nashm_224x224_nv12.hbm`） |
| `s600/mobilenetv1_100_nashp_224x224_nv12.hbm` | s600 | 100 | 单阶段 | 下载（`s:mobilenetv1:s600/mobilenetv1_100_nashp_224x224_nv12.hbm`） |
| `mobilenetv1_125_bayese_224x224_nv12.bin` | x5 | 125 | 单阶段 | 下载（`x5:mobilenetv1:mobilenetv1_125_bayese_224x224_nv12.bin`） |
| `s100/mobilenetv1_125_nashe_224x224_nv12.hbm` | s100 | 125 | 单阶段 | 下载（`s:mobilenetv1:s100/mobilenetv1_125_nashe_224x224_nv12.hbm`） |
| `s100p/mobilenetv1_125_nashm_224x224_nv12.hbm` | s100p | 125 | 单阶段 | 下载（`s:mobilenetv1:s100p/mobilenetv1_125_nashm_224x224_nv12.hbm`） |
| `s600/mobilenetv1_125_nashp_224x224_nv12.hbm` | s600 | 125 | 单阶段 | 下载（`s:mobilenetv1:s600/mobilenetv1_125_nashp_224x224_nv12.hbm`） |

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
bash samples/vision/mobilenetv1/model/download.sh x5 100
bash samples/vision/mobilenetv1/model/download.sh x5 125
bash samples/vision/mobilenetv1/model/download.sh s100p 125
```

Python 形式等价：
`python3 samples/vision/mobilenetv1/model/download.py --target s100 --variant 100`。下载器先写同目录临时文件，按 Manifest 中的
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
| `mobilenetv1_100_bayese_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入 (224x224), F32 `[1,1000,1,1]` logits 输出 | `23392a747c6c50b97f2ffe7412b0e2b3ded9245355fe158b988712007d501056` |
| `s100/mobilenetv1_100_nashe_224x224_nv12.hbm` | nash-e `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `8a27a19dbe4d9878328bb928aa8af395367b64a688629e82388e55410f25472d` |
| `s100p/mobilenetv1_100_nashm_224x224_nv12.hbm` | nash-m `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `34ba7820078612d9e070c601cedc9006345353547cbb4c6abdda991bdf1b06b3` |
| `s600/mobilenetv1_100_nashp_224x224_nv12.hbm` | nash-p `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `160c05acb159ae6170786bab32be42305ed62613468f45828f4f7fbe3939532b` |
| `mobilenetv1_125_bayese_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入 (224x224), F32 `[1,1000,1,1]` logits 输出 | `c4e3ef8bfcd41d25ca7c8e743cf443eba0841187eca78d737cb15d71ae53af80` |
| `s100/mobilenetv1_125_nashe_224x224_nv12.hbm` | nash-e `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `da510f1184db1b12cba0f9861f92133a5bede5eaf1088e4b7696813de9be205e` |
| `s100p/mobilenetv1_125_nashm_224x224_nv12.hbm` | nash-m `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `00f10280df4376f0de9f2c440fb70444d66155b24145ddff86c0b66102e07718` |
| `s600/mobilenetv1_125_nashp_224x224_nv12.hbm` | nash-p `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `993f487a9861a7281479d9a0cf825a253df7752cd2bb3ec3ff5dc2509bcfff64` |

所有制品都是对固定版本 [timm](https://github.com/huggingface/pytorch-image-models)
检查点（`mobilenetv1_100.ra4_e3600_r224_in1k`、`mobilenetv1_125.ra4_e3600_r224_in1k`，Apache-2.0）做训练后 INT8 量化的结果，
并按文件名中的 march 标记编译：`bayese`（bayes-e，X5）、`nashe`（nash-e，
S100）、`nashm`（nash-m，S100P）、`nashp`（nash-p，S600）。网络输入的归一化
已编译进模型；runtime 送入的是由模型输入尺寸的中心裁剪图转换的 NV12
（见[转换说明](../conversion/README_cn.md#preprocessing)）。
`100` 的 S100、S100P、S600 构建额外使用工具链的权重偏差校正（仍为 INT8，见
[转换说明](../conversion/README_cn.md#compile)）。
