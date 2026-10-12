[English](README.md) | 简体中文

# ResNet 模型制品

model 目录不提交任何二进制；制品由 下载器按平台发布 Manifest
显式获取。

<a id="artifacts"></a>
## 制品清单

| 文件 | Target | 阶段 | 来源 |
| --- | --- | --- | --- |
| `resnet18_224x224_nv12.bin` | x5 | 单阶段 | 下载（`x5:resnet:resnet18_224x224_nv12.bin`） |
| `s100/resnet18_224x224_nv12.hbm` | s100 | 单阶段 | 下载（`s:resnet18:s100/resnet18_224x224_nv12.hbm`） |
| `s600/resnet18_224x224_nv12.hbm` | s600 | 单阶段 | 下载（`s:resnet18:s600/resnet18_224x224_nv12.hbm`） |
| `s100/resnet50_224x224_nv12.hbm` | s100 | 单阶段 | 下载（`s:resnet50:s100/resnet50_224x224_nv12.hbm`） |
| `s600/resnet50_224x224_nv12.hbm` | s600 | 单阶段 | 下载（`s:resnet50:s600/resnet50_224x224_nv12.hbm`） |
| `s100/resnet152_224x224_nv12.hbm` | s100 | 单阶段 | 下载（`s:resnet152:s100/resnet152_224x224_nv12.hbm`） |
| `s600/resnet152_224x224_nv12.hbm` | s600 | 单阶段 | 下载（`s:resnet152:s600/resnet152_224x224_nv12.hbm`） |

每个引用对应 `docs/release/x5/models.yaml` 或
`docs/release/s/models.yaml` 中的一行；URL 与格式以 Manifest 为
权威。X5 消费一个 packed NV12 张量；S100/S600 消费分离的 Y 与 UV 张量——
仅凭文件名无法确定协议，因此 runtime 始终将 `--model-path` 与完整引用
配对。S100P 没有 ResNet 行，复用 S100 文件不能使其有效。

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

在仓库根目录执行所需板卡和变体的命令：

```bash
bash samples/vision/resnet/model/download.sh x5 resnet18
bash samples/vision/resnet/model/download.sh s100 resnet18
bash samples/vision/resnet/model/download.sh s100 resnet50
bash samples/vision/resnet/model/download.sh s100 resnet152
bash samples/vision/resnet/model/download.sh s600 resnet18
bash samples/vision/resnet/model/download.sh s600 resnet50
bash samples/vision/resnet/model/download.sh s600 resnet152
```

Shell 位置参数依次为板卡和变体；省略变体时选择 `resnet18`。Python 入口使用 `--target` 和 `--variant`。下载文件按清单文件名保存在本目录，S 系列使用对应板卡子目录。命令打印保存路径和 SHA-256。

<a id="accompanying-files"></a>
## 伴随文件

分类器运行还需要逐行一个类别的 ImageNet 标签文件：
`datasets/imagenet/imagenet_classes.names`（X5 与 S 系列共用；原各平台
快照副本内容一致）。该文件在仓库内，无需下载。

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
| `s100/resnet50_224x224_nv12.hbm` | nash `.hbm`，split Y/UV 输入，F32 `[1,1000]` logits 输出 | null (unknown) |
| `s600/resnet50_224x224_nv12.hbm` | nash `.hbm`，split Y/UV 输入，F32 `[1,1000]` logits 输出 | null (unknown) |
| `s100/resnet152_224x224_nv12.hbm` | nash `.hbm`，split Y/UV 输入，F32 `[1,1000]` logits 输出 | null (unknown) |
| `s600/resnet152_224x224_nv12.hbm` | nash `.hbm`，split Y/UV 输入，F32 `[1,1000]` logits 输出 | null (unknown) |

这些 Manifest 行的 SHA-256 为 `null (unknown)`。下载器打印下载文件的 SHA-256，供本地记录。
