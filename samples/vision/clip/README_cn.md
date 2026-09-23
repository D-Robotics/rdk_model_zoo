[English](./README.md) | 简体中文

# CLIP 图文匹配

<a id="overview"></a>
## 算法与来源

CLIP 将图像和候选文本映射到共享的 512 维空间，并用 cosine similarity 对文本排序。本 sample 保留源制品边界：图像 encoder 是 X5 BPU `.bin` 模型，文本 encoder 是 CPU ONNX 模型，源 BPE 词表保留在 `runtime/python/bpe_simple_vocab_16e6.txt.gz`。源 commit 为 `ac115717197920355fc390bb04299b20e6436864`。

维护的 task 只有三个阶段：`pre_process` 将一张 BGR 图片和 prompt 列表转换为图像/token tensor，`forward` 运行两个 encoder 并返回原始特征，`post_process` 计算 cosine 分数及降序排列；另提供 `predict`。绘图由独立 helper 负责。

<a id="support-matrix"></a>
## 支持与实测矩阵

active manifest 只有 RDK X5 的制品对。没有 S100、S100P、S600 的 CLIP 制品，也未提供 C++ runtime。板端状态不代表本轮已实测板卡。

| Variant | x5 | s100 | s100p | s600 | Python | C++ |
| --- | --- | --- | --- | --- | --- | --- |
| `clip-image-text-pair` | supported-not-run | not-supported | not-supported | not-supported | supported-not-run | not-supported |

板端验证证据：not-run。主机测试使用注入的 image/ONNX fixture 和源 tokenizer 对照，不代表 X5 执行验证。

<a id="prerequisites"></a>
## 环境前提

- 板端：RDK X5 镜像，需提供图像 encoder 的 `hbm_runtime` 和 CPU 文本 encoder 的 `onnxruntime`。镜像与固件版本未核验。
- 主机检查：Python 3.14.7、`numpy`、`opencv-python`、`PyYAML`、`ftfy==6.3.1`、`regex==2026.9.10`；主机测试不需要 ONNX Runtime。
- 板端推理还需 `onnxruntime`、两个模型制品和内置 BPE 词表。

<a id="quickstart"></a>
## 快速体验

显式准备两个模型，再从仓库根目录在 X5 上运行。当前 `run.sh` 不会自动下载，这是相对历史源 launcher 的有意行为差异。

```bash
# cwd：仓库根目录；来源：docs/release/x5/models.yaml 中的精确 URL
python3 samples/vision/clip/model/download.py --target x5
# 预期：samples/vision/clip/model/img_encoder.bin 和 text_encoder.onnx

# cwd：仓库根目录；输入：test_data/dog.jpg；prompt：默认 a diagram,a dog
python3 samples/vision/clip/runtime/python/main.py --target x5
# 预期：打印 JSON prompts/scores/order，并生成 samples/vision/clip/test_data/inference.png；退出码 0
```

默认可视化路径是受版本控制的 `test_data/inference.png`，运行会覆盖它。传入 `--img-save-path` 可保存到其他位置。`run.sh` 只是快捷入口，不准备模型。

<a id="expected-results"></a>
## 预期结果

CLI 打印 `target`、`prompts`、`scores`、`order`、`image_saved`。`scores` 按 prompt 原顺序保存 cosine similarity，`order` 是降序 prompt 索引。绘图会将每个 prompt 和分数写入输入图片的副本。源评估预期 dog 图片对 `a dog` 的分数高于 `a diagram`；没有公开数值 benchmark。

<a id="directory"></a>
## 目录职责

```text
clip/
├── conversion/             # 图像/文本协议与转换边界
├── evaluator/              # 验证条件；没有公开 benchmark
├── model/                  # 基于 manifest 的双模型准备
├── runtime/python/         # BPE、预处理、双 encoder runner、task、CLI、绘图
├── test_data/              # dog.jpg 和 inference.png
└── README.md               # 英文说明
```

<a id="entry-points"></a>
## 入口索引

- 模型准备：[`model/README_cn.md`](model/README_cn.md) —— X5 图像 `.bin`、文本 `.onnx`，分别使用独立 asset identity。
- Python 运行：[`runtime/python/README_cn.md`](runtime/python/README_cn.md) —— BPE tokenizer、BPU 图像 + CPU ONNX 文本推理、cosine 排序和绘图。
- C++ 运行：未提供；C++ 为 `not-supported`。
- 模型转换：[`conversion/README_cn.md`](conversion/README_cn.md) —— 源协议和缺失的导出/校准配方。
- 模型评估：[`evaluator/README_cn.md`](evaluator/README_cn.md) —— 源验证路径，不虚构 benchmark 数值。

<a id="license"></a>
## 许可说明

示例代码遵循仓库 [LICENSE](../../../LICENSE) 的 Apache-2.0。源 CLIP 制品沿用 X5 发布记录中的许可证和来源；本 sample 不新增模型许可证断言。保留源迁移记录中的贡献者信息。
