# MobileNetV1 模型制品

model 目录不提交任何二进制；制品由 canonical 下载器按平台发布 Manifest
显式获取。

<a id="artifacts"></a>
## 制品清单

| 文件 | Target | 变体 | 阶段 | 来源 |
| --- | --- | --- | --- | --- |
| `mobilenetv1_224x224_nv12.bin` | x5 | mobilenetv1 | 单阶段 | 下载（`x5:mobilenetv1:mobilenetv1_224x224_nv12.bin`） |
| `s100/mobilenetv1_224x224_nv12.hbm` | s100 | mobilenetv1 | 单阶段 | 下载（`s:mobilenetv1:s100/mobilenetv1_224x224_nv12.hbm`） |
| `s600/mobilenetv1_224x224_nv12.hbm` | s600 | mobilenetv1 | 单阶段 | 下载（`s:mobilenetv1:s600/mobilenetv1_224x224_nv12.hbm`） |

每个引用对应 `docs/release/x5/models.yaml` 或 `docs/release/s/models.yaml`
中的一行；URL 与格式以 Manifest 为权威。X5 消费一个 packed NV12 张量；
S100/S600 消费分离的 Y 与 UV 张量——仅凭文件名无法确定协议，因此 runtime
始终将 `--model-path` 与完整引用配对。S100P 没有对应资产行，复用 S100
文件不能使其有效。

<a id="preparation"></a>
## 准备步骤

在仓库根目录执行：

```bash
# 输入：目标对应的 Manifest 行 — 输出：本目录下的制品文件
# 成功判据：退出码 0，打印观察 digest；不留半成品文件
bash samples/vision/mobilenetv1/model/download.sh x5    # 或 s100 / s600
```

Python 形式等价：
`python3 samples/vision/mobilenetv1/model/download.py --target s100`。下载器先写同目录临时文件，存在发布者
SHA-256 时校验，原子落盘且不覆盖已有文件（校验失败时保留原文件供调查）。
当前行没有发布者 SHA-256，因此下载器打印观察 digest 作为本地证据并说明
来源未被证明。下载是显式操作，推理过程不会触发下载。

<a id="accompanying-files"></a>
## 伴随文件

分类器运行还需要逐行一个类别的 ImageNet 标签文件：
`datasets/imagenet/imagenet_classes.names`（X5 与 S 系列共用）。该文件在
仓库内，无需下载。`test_data/` 内源分支带来的标签副本仅为遗留物，canonical
路径以根目录 `datasets/` 为准。

<a id="local-paths"></a>
## 本地路径

准备完成后，制品位于 sample 的 `model/` 目录（X5 平铺，S 系列在
`model/s100/`、`model/s600/` 子目录），相对 sample 根目录；
[runtime/python/README_cn.md](../runtime/python/README_cn.md) 中的
`--model-path` 示例即指向这些位置。

<a id="formats-checksums"></a>
## 格式与校验值

| 文件 | 格式 | SHA-256 |
| --- | --- | --- |
| `mobilenetv1_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入, F32 `[1,1000,1,1]` 概率输出 | null (unknown) |
| `s100/mobilenetv1_224x224_nv12.hbm` | nash `.hbm`，split Y/UV 输入, F32 `[1,1000]` 概率输出 | null (unknown) |
| `s600/mobilenetv1_224x224_nv12.hbm` | nash `.hbm`，split Y/UV 输入, F32 `[1,1000]` 概率输出 | null (unknown) |

这些 Manifest 行未记录发布者 SHA-256；未知值保持 `null (unknown)`，禁止
跨制品复制。每次下载都会打印观察 digest 作为本地证据。
