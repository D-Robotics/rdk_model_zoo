[English](README.md) | 简体中文

# MobileNetV2 模型制品

model 目录不提交任何二进制；制品由 下载器按平台发布 Manifest
显式获取。

<a id="artifacts"></a>
## 制品清单

| 文件 | Target | 变体 | 阶段 | 来源 |
| --- | --- | --- | --- | --- |
| `mobilenetv2_100_bayese_224x224_nv12.bin` | x5 | 100 | 单阶段 | 下载（`x5:mobilenetv2:mobilenetv2_100_bayese_224x224_nv12.bin`） |
| `s100/mobilenetv2_100_nashe_224x224_nv12.hbm` | s100 | 100 | 单阶段 | 下载（`s:mobilenetv2:s100/mobilenetv2_100_nashe_224x224_nv12.hbm`） |
| `s100p/mobilenetv2_100_nashm_224x224_nv12.hbm` | s100p | 100 | 单阶段 | 下载（`s:mobilenetv2:s100p/mobilenetv2_100_nashm_224x224_nv12.hbm`） |
| `s600/mobilenetv2_100_nashp_224x224_nv12.hbm` | s600 | 100 | 单阶段 | 下载（`s:mobilenetv2:s600/mobilenetv2_100_nashp_224x224_nv12.hbm`） |
| `mobilenetv2_140_bayese_224x224_nv12.bin` | x5 | 140 | 单阶段 | 下载（`x5:mobilenetv2:mobilenetv2_140_bayese_224x224_nv12.bin`） |
| `s100/mobilenetv2_140_nashe_224x224_nv12.hbm` | s100 | 140 | 单阶段 | 下载（`s:mobilenetv2:s100/mobilenetv2_140_nashe_224x224_nv12.hbm`） |
| `s100p/mobilenetv2_140_nashm_224x224_nv12.hbm` | s100p | 140 | 单阶段 | 下载（`s:mobilenetv2:s100p/mobilenetv2_140_nashm_224x224_nv12.hbm`） |
| `s600/mobilenetv2_140_nashp_224x224_nv12.hbm` | s600 | 140 | 单阶段 | 下载（`s:mobilenetv2:s600/mobilenetv2_140_nashp_224x224_nv12.hbm`） |

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
bash samples/vision/mobilenetv2/model/download.sh x5 100
bash samples/vision/mobilenetv2/model/download.sh x5 140
bash samples/vision/mobilenetv2/model/download.sh s100p 140
```

Python 形式等价：
`python3 samples/vision/mobilenetv2/model/download.py --target s100 --variant 100`。下载器先写同目录临时文件，按 Manifest 中的
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
| `mobilenetv2_100_bayese_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入 (224x224), F32 `[1,1000,1,1]` logits 输出 | `e044c3acb08403f2a3aee6342f4f131b798496773ebc7e4ae34a623ecff3e242` |
| `s100/mobilenetv2_100_nashe_224x224_nv12.hbm` | nash-e `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `e31d0445ead5f361dc0dae6a4c959d671a9c878a4f5a4d508d7fd5b9cc6672b5` |
| `s100p/mobilenetv2_100_nashm_224x224_nv12.hbm` | nash-m `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `906082fa0df94c88ad00187b6120f802a9e2047fc9a33745f737516e8f634159` |
| `s600/mobilenetv2_100_nashp_224x224_nv12.hbm` | nash-p `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `2509ba5e2a47ae97a7a43cc83fbb60c03b0cc5899f4580e8b4cc82ce38eafe1b` |
| `mobilenetv2_140_bayese_224x224_nv12.bin` | bayes-e `.bin`，packed NV12 输入 (224x224), F32 `[1,1000,1,1]` logits 输出 | `bf2a8920efbd7382e0fd63841d6217f9d0c311ea96cbaeee737d07dfe62e5aa9` |
| `s100/mobilenetv2_140_nashe_224x224_nv12.hbm` | nash-e `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `77155795844def681c4951ea4ee7e229a3cf1b6efd7ef8c71a0678731e189085` |
| `s100p/mobilenetv2_140_nashm_224x224_nv12.hbm` | nash-m `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `887d47c7825e1829177ffdf35f2b2b8d895b7d8c02f0fa1e1f9e2b09308ce63f` |
| `s600/mobilenetv2_140_nashp_224x224_nv12.hbm` | nash-p `.hbm`，split Y/UV 输入 (224x224), F32 `[1,1000]` logits 输出 | `b868f6aacfa7a8380f7950458df7da5a240d67d2a2f6c2dec7e08785f631cd60` |

所有制品都是对固定版本 [timm](https://github.com/huggingface/pytorch-image-models)
检查点（`mobilenetv2_100.ra_in1k`、`mobilenetv2_140.ra_in1k`，Apache-2.0）做训练后 INT8 量化的结果，
并按文件名中的 march 标记编译：`bayese`（bayes-e，X5）、`nashe`（nash-e，
S100）、`nashm`（nash-m，S100P）、`nashp`（nash-p，S600）。网络输入的归一化
已编译进模型；runtime 送入的是由模型输入尺寸的中心裁剪图转换的 NV12
（见[转换说明](../conversion/README_cn.md#preprocessing)）。
