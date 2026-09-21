# FastViT 模型制品

model 目录不检入任何二进制；制品由规范化下载器按平台发布 Manifest
显式获取。

<a id="artifacts"></a>
## 制品

| 文件 | Target | 变体 | 阶段 | 来源 |
| --- | --- | --- | --- | --- |
| `FastViT_S12_224x224_nv12.bin` | x5 | S | single | download（`x5:fastvit:FastViT_S12_224x224_nv12.bin`） |
| `FastViT_SA12_224x224_nv12.bin` | x5 | T0 | single | download（`x5:fastvit:FastViT_SA12_224x224_nv12.bin`） |
| `FastViT_T12_224x224_nv12.bin` | x5 | T1 | single | download（`x5:fastvit:FastViT_T12_224x224_nv12.bin`） |
| `FastViT_T8_224x224_nv12.bin` | x5 | T2 | single | download（`x5:fastvit:FastViT_T8_224x224_nv12.bin`） |

每条引用都是 `docs/release/x5/models.yaml` 中的一行；Manifest 是 URL 与
格式的事实源。X5 消费单个 packed NV12 张量，因此运行时总是把
`--model-path` 与精确引用配对使用。本模型没有 S 系列制品：S100/S100P/
S600 的选择是显式的 no-published-asset 错误，也不得在 S 板卡上复用 X5
文件。

<a id="preparation"></a>
## 准备

在仓库根目录执行：

```bash
# 输入：目标/变体对应的 Manifest 行 — 输出：本目录下的文件
# 成功判据：退出码 0，打印实测摘要；不残留半成品文件
bash samples/vision/fastvit/model/download.sh x5 S
bash samples/vision/fastvit/model/download.sh x5 T2
```

Python 形式等价：
`python3 samples/vision/fastvit/model/download.py --target x5 --variant T1`。
下载器经同目录临时文件写入，核对内容长度与 Manifest 记录的发布方
SHA-256（如有），原子落盘且不覆盖已有文件（校验失败保留文件供排查）。
当前各行没有发布方 SHA-256，下载器打印实测摘要作为本地证据并声明来源
未经独立证明。下载是显式动作，绝不发生在推理过程中。源 `download.sh`
用 `wget` 拉取两个文件且不做任何校验；该路径已被本 Manifest 驱动下载
取代。

<a id="accompanying-files"></a>
## 随伴文件

分类运行还需要每行一个类别的 ImageNet 标签文件：
`datasets/imagenet/imagenet_classes.names`。该文件已检入仓库，无需下载。

<a id="local-paths"></a>
## 本地路径

准备完成后，制品平铺在 sample 的 `model/` 目录下，相对 sample 根；
[runtime/python/README_cn.md](../runtime/python/README_cn.md) 中的
`--model-path` 示例指向这些位置。

<a id="formats-checksums"></a>
## 格式与校验值

| 文件 | 格式 | SHA-256 |
| --- | --- | --- |
| `FastViT_S12_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入（224x224），F32 `[1,1000,1,1]` logits 输出 | null（未知） |
| `FastViT_SA12_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入（224x224），F32 `[1,1000,1,1]` logits 输出 | null（未知） |
| `FastViT_T12_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入（224x224），F32 `[1,1000,1,1]` logits 输出 | null（未知） |
| `FastViT_T8_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入（224x224），F32 `[1,1000,1,1]` logits 输出 | null（未知） |

Manifest 未记录这些行的发布方 SHA-256；未知值保持 `null（未知）`，
绝不在制品间复制。下载器每次下载都打印实测摘要作为本地证据。
