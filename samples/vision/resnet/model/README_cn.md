# ResNet18 模型制品

model 目录不提交任何二进制；制品由 canonical 下载器按平台发布 Manifest
显式获取。

<a id="artifacts"></a>
## 制品清单

| 文件 | Target | 阶段 | 来源 |
| --- | --- | --- | --- |
| `resnet18_224x224_nv12.bin` | x5 | 单阶段 | 下载（`x5:resnet:resnet18_224x224_nv12.bin`） |
| `s100/resnet18_224x224_nv12.hbm` | s100 | 单阶段 | 下载（`s:resnet18:s100/resnet18_224x224_nv12.hbm`） |
| `s600/resnet18_224x224_nv12.hbm` | s600 | 单阶段 | 下载（`s:resnet18:s600/resnet18_224x224_nv12.hbm`） |

每个引用对应 `platforms/x5/docs/release/models.yaml` 或
`platforms/s/docs/release/models.yaml` 中的一行；URL 与格式以 Manifest 为
权威。X5 消费一个 packed NV12 张量；S100/S600 消费分离的 Y 与 UV 张量——
仅凭文件名无法确定协议，因此 runtime 始终将 `--model-path` 与完整引用
配对。S100P 没有 ResNet18 行，复用 S100 文件不能使其有效。

<a id="preparation"></a>
## 准备步骤

在仓库根目录执行：

```bash
# 输入：目标对应的 Manifest 行 — 输出：本目录下的制品文件
# 成功判据：退出码 0，打印观察 digest；不留半成品文件
bash samples/vision/resnet/model/download.sh x5    # 或 s100 / s600
```

Python 形式等价：
`python3 samples/vision/resnet/model/download.py --target s100`。下载器先写
同目录临时文件，存在发布者 SHA-256 时校验，原子落盘且不覆盖已有文件
（校验失败时保留原文件供调查）。当前行没有发布者 SHA-256，因此下载器打印
观察 digest 作为本地证据并说明来源未被证明。下载是显式操作，推理过程不会
触发下载。

<a id="accompanying-files"></a>
## 伴随文件

分类器运行还需要逐行一个类别的 ImageNet 标签文件：
`platforms/x5/datasets/imagenet/imagenet_classes.names`（X5）或
`platforms/s/datasets/imagenet/imagenet_classes.names`（S 系列）。两者都在
仓库内，无需下载。

<a id="local-paths"></a>
## 本地路径

准备完成后，制品位于 sample 根目录相对路径 `model/resnet18_224x224_nv12.bin`
（x5）、`model/s100/resnet18_224x224_nv12.hbm`（s100）、
`model/s600/resnet18_224x224_nv12.hbm`（s600）；
[runtime/python/README_cn.md](../runtime/python/README_cn.md) 中的
`--model-path` 示例即指向这些位置，C++ 启动器默认也使用 `model/s100`。

<a id="formats-checksums"></a>
## 格式与校验值

| 文件 | 格式 | SHA-256 |
| --- | --- | --- |
| `resnet18_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入，F32 `[1,1000,1,1]` 输出 | null (unknown) |
| `s100/resnet18_224x224_nv12.hbm` | nash `.hbm`，split Y/UV 输入，F32 `[1,1000]` 输出 | null (unknown) |
| `s600/resnet18_224x224_nv12.hbm` | nash `.hbm`，split Y/UV 输入，F32 `[1,1000]` 输出 | null (unknown) |

这些 Manifest 行未记录发布者 SHA-256；未知值保持 `null (unknown)`，禁止
跨制品复制。每次下载都会打印观察 digest 作为本地证据。
