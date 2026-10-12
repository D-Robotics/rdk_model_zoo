[English](README.md) | 简体中文

# MobileNetV3 模型制品

model 目录不提交任何二进制；制品由 下载器按平台发布 Manifest
显式获取。

<a id="artifacts"></a>
## 制品清单

| 文件 | Target | 变体 | 阶段 | 来源 |
| --- | --- | --- | --- | --- |
| `s100/mobilenetv3_large_100_nashe_224x224_nv12.hbm` | s100 | large | 单阶段 | 下载（`s:mobilenetv3:s100/mobilenetv3_large_100_nashe_224x224_nv12.hbm`） |
| `s100p/mobilenetv3_large_100_nashm_224x224_nv12.hbm` | s100p | large | 单阶段 | 下载（`s:mobilenetv3:s100p/mobilenetv3_large_100_nashm_224x224_nv12.hbm`） |
| `s600/mobilenetv3_large_100_nashp_224x224_nv12.hbm` | s600 | large | 单阶段 | 下载（`s:mobilenetv3:s600/mobilenetv3_large_100_nashp_224x224_nv12.hbm`） |
| `s100/mobilenetv3_small_100_nashe_224x224_nv12.hbm` | s100 | small | 单阶段 | 下载（`s:mobilenetv3:s100/mobilenetv3_small_100_nashe_224x224_nv12.hbm`） |
| `s100p/mobilenetv3_small_100_nashm_224x224_nv12.hbm` | s100p | small | 单阶段 | 下载（`s:mobilenetv3:s100p/mobilenetv3_small_100_nashm_224x224_nv12.hbm`） |
| `s600/mobilenetv3_small_100_nashp_224x224_nv12.hbm` | s600 | small | 单阶段 | 下载（`s:mobilenetv3:s600/mobilenetv3_small_100_nashp_224x224_nv12.hbm`） |

每个引用对应 `docs/release/s/models.yaml`
中的一行；URL 与格式以 Manifest 为权威。
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
bash samples/vision/mobilenetv3/model/download.sh s100 large
bash samples/vision/mobilenetv3/model/download.sh s100 small
bash samples/vision/mobilenetv3/model/download.sh s100p small
```

Python 形式等价：
`python3 samples/vision/mobilenetv3/model/download.py --target s100 --variant large`。下载器先写同目录临时文件，按 Manifest 中的
SHA-256 校验，原子落盘且不覆盖已有文件（校验失败时保留原文件供调查）。
下载后打印的 digest 应与下表中的 SHA-256 一致。下载是显式操作，推理过程不会
触发下载。

<a id="accompanying-files"></a>
## 伴随文件

分类器运行使用 ImageNet 标签文件
`datasets/imagenet/imagenet_classes.names`；运行时
`--label-file` 的默认值即该路径，文件在仓库内，无需下载。`test_data/`
内的 `imagenet_classes.names` 内容与该文件相同，需要时经 `--label-file`
显式指定。

<a id="local-paths"></a>
## 本地路径

准备完成后，制品位于 sample 的 `model/` 目录（S 系列在
`model/s100/`、`model/s100p/`、`model/s600/` 子目录），相对 sample 根目录；
[runtime/python/README_cn.md](../runtime/python/README_cn.md) 中的
`--model-path` 示例即指向这些位置。

<a id="formats-checksums"></a>
## 格式与校验值

| 文件 | 格式 | SHA-256 |
| --- | --- | --- |
| `s100/mobilenetv3_large_100_nashe_224x224_nv12.hbm` | nash-e `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `9190d5334b2f1488bbfa3cc4f918bb21ec788d4fcc08a55074b5287cb7baf24d` |
| `s100p/mobilenetv3_large_100_nashm_224x224_nv12.hbm` | nash-m `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `96e6bda58009330c96d139bdf858326ddf5481299b68d6ecadd3b2b3bc7a1658` |
| `s600/mobilenetv3_large_100_nashp_224x224_nv12.hbm` | nash-p `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `9b2b2422566214ebdbad27cedebbbaf5668edd9272c3efc51adb1113be8dbad4` |
| `s100/mobilenetv3_small_100_nashe_224x224_nv12.hbm` | nash-e `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `c8cfa5cfe34d46b99e235d9823bbbf464e67808d59d9e3281593c87cc07bd6fc` |
| `s100p/mobilenetv3_small_100_nashm_224x224_nv12.hbm` | nash-m `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `a67df2c4154d44c9ebe25e5250d16409a9a1ad31c953cc311b703f396151bcec` |
| `s600/mobilenetv3_small_100_nashp_224x224_nv12.hbm` | nash-p `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `8f2177eacf3496aeea1c4234146ea73eaad99882ff74b0bd0afa47b266849b08` |

所有制品都是对固定版本 [timm](https://github.com/huggingface/pytorch-image-models)
检查点（`mobilenetv3_large_100.ra_in1k`、`mobilenetv3_small_100.lamb_in1k`，Apache-2.0）做训练后量化（INT16 激活、INT8 权重）的结果，
并按文件名中的 march 标记编译：`nashe`（nash-e，
S100）、`nashm`（nash-m，S100P）、`nashp`（nash-p，S600）。网络输入的归一化
已编译进模型；runtime 送入的是由模型输入尺寸的中心裁剪图转换的 NV12
（见[转换说明](../conversion/README_cn.md#preprocessing)）。
