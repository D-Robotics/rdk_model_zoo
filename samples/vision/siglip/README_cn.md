[English](./README.md) | 简体中文

# SigLIP 视觉特征

<a id="overview"></a>
## 算法与来源

SigLIP 视觉编码器把一张图片转换为全局嵌入或 patch 特征序列。本 sample 只提供 SigLIP 的视觉侧，不包含文本编码器和文本 token 流程。上游论文是 [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)，模型家族由 [Google Research](https://github.com/google-research/big_vision) 发布；本 sample 位于 `samples/vision/siglip`。

八个已发布 variant 均为打包 HBM 制品。每个制品包含固定的 `pooler_output` 和 `last_hidden_state` 两个子模型，使用同一图像输入，并且每次只执行所选子模型。

<a id="support-matrix"></a>
## 支持与实测矩阵

下表描述发布支持，不表示本轮已连接板卡。八个 variant 的两个子模型在 S100、S100P 上均为 `supported-not-run`；文件名中的 `s100/` 是共享发布制品路径，不会删去 S100P 支持。Python 为 `supported-not-run`；没有 C++ 实现，因此 C++ 为 `not-supported`。X5、S600 为 `not-supported`。

| Variant | x5 | s100 | s100p | s600 | Python | C++ |
| --- | --- | --- | --- | --- | --- | --- |
| `base-patch16-224` | not-supported | supported-not-run | supported-not-run | not-supported | supported-not-run | not-supported |
| `base-patch16-384` | not-supported | supported-not-run | supported-not-run | not-supported | supported-not-run | not-supported |
| `base-patch16-512` | not-supported | supported-not-run | supported-not-run | not-supported | supported-not-run | not-supported |
| `large-patch16-256` | not-supported | supported-not-run | supported-not-run | not-supported | supported-not-run | not-supported |
| `large-patch16-384` | not-supported | supported-not-run | supported-not-run | not-supported | supported-not-run | not-supported |
| `so400m-patch14-224` | not-supported | supported-not-run | supported-not-run | not-supported | supported-not-run | not-supported |
| `so400m-patch14-384` | not-supported | supported-not-run | supported-not-run | not-supported | supported-not-run | not-supported |
| `so400m-patch16-256-i18n` | not-supported | supported-not-run | supported-not-run | not-supported | supported-not-run | not-supported |

板端验证证据：not-run。主机测试只覆盖契约和注入 fixture，不代表 `hbm_runtime` 已在板端运行。

<a id="prerequisites"></a>
## 环境前提

- 板卡：RDK S100 或 S100P，板端镜像必须提供 `hbm_runtime`；本轮未核验镜像和 BPU 固件版本。
- 主机准备：Python 3.14.7，以及 `requirements-host.txt` 中的 `numpy`、`opencv-python`、`PyYAML`。
- 推理前必须准备 HBM；运行时命令不会隐式下载模型。
- 源资料没有给出除 HBM 文件外的磁盘或内存要求；板端资源未实测。

<a id="quickstart"></a>
## 快速体验

下面是一条完整的显式路径。第一条命令需要网络，第二条命令需要本地 S100/S100P 板卡；本轮文档迁移没有执行下载。

```bash
# cwd：仓库根目录；来源：model/README.md 中的发布 manifest URL
python3 samples/vision/siglip/model/download.py --target s100 --variant base-patch16-224
# 预期：samples/vision/siglip/model/s100/bpu-siglip-base-patch16-224.hbm

# cwd：仓库根目录；输入：samples/vision/siglip/test_data/dog.jpg
python3 samples/vision/siglip/runtime/python/main.py --target s100 --variant base-patch16-224 --submodel pooler_output
# 预期：打印包含 submodel、shape、dtype、mean、std、min、max、l2_norm 的 JSON；退出码 0
```

快捷脚本 `runtime/python/run.sh` 兼容历史位置参数（`pooler_output` 或 `last_hidden_state`），其后可追加命名参数；不会下载模型。

<a id="expected-results"></a>
## 预期结果

CLI 为所选原始特征 tensor 打印一个 JSON 统计对象。`shape` 来自 HBM metadata：`pooler_output` 为 `(1,D)` 或 `(1,1,D)`，`last_hidden_state` 为 `(1,N,D)`；`D`、`N` 见 runtime README。保留制品原生 dtype；运行时不会反量化、softmax、归一化、squeeze 或其他数值变换。板端运行制品前，不声明具体数值。

<a id="directory"></a>
## 目录职责

```text
siglip/
├── conversion/     # HBM 转换配方边界与缺失项
├── evaluator/      # 历史性能/精度表及对照流程
├── model/          # 基于 manifest 的 HBM 准备脚本
├── runtime/python/ # Python binding、runner、task 和 CLI
├── test_data/      # dog.jpg fixture
├── README.md       # 英文说明
└── README_cn.md    # 本文件
```

<a id="entry-points"></a>
## 入口索引

- 模型准备：[`model/README_cn.md`](model/README_cn.md) —— 八个由 manifest 管理、同时支持 S100/S100P 的 HBM 制品。
- Python 运行：[`runtime/python/README_cn.md`](runtime/python/README_cn.md) —— 预处理、metadata binding、选定子模型运行和 JSON 摘要 CLI。
- C++ 运行：未提供；C++ 为 `not-supported`。
- 模型转换：[`conversion/README_cn.md`](conversion/README_cn.md) —— 源信息和不可复现边界。
- 模型评估：[`evaluator/README_cn.md`](evaluator/README_cn.md) —— 历史表格以及不执行推理的 legacy/unified 对照流程。

<a id="license"></a>
## 许可说明

示例代码遵循仓库 [LICENSE](../../../LICENSE) 的 Apache-2.0。manifest 指向 Google 来源的预编译 SigLIP 制品，但源发布记录没有记录每个权重/导出制品的具体许可证和版本，因此这里不额外断言模型许可证。保留源贡献者署名：Cauchy @吴超。
