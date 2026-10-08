[English](./README.md) | 简体中文

# CLIP 图文匹配

<a id="overview"></a>
## 算法与来源

CLIP 学习图像与文本的共同表示。本示例使用 BPU 图像编码器和 CPU ONNX 文本编码器，比较图片与候选描述，并按余弦相似度对描述排序。

<a id="directory"></a>
## 目录结构

```text
clip/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── tests/  # 自动化测试
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── requirements-host.txt  # 源码或数据文件
```

<a id="support-matrix"></a>
## 支持与实测矩阵

active manifest 只有 RDK X5 的制品对。没有 S100、S100P、S600 的 CLIP 制品，也未提供 C++ runtime。

| Variant | x5 | s100 | s100p | s600 | Python | C++ |
| --- | --- | --- | --- | --- | --- | --- |
| `clip-image-text-pair` | supported | not-supported | not-supported | not-supported | supported | not-supported |

板端执行需要安装 `hbm_runtime` 和 `onnxruntime` 的 X5 板卡，运行前准备图像与文本模型。

<a id="prerequisites"></a>
## 环境前提

- 板端：RDK X5 镜像，需提供图像 encoder 的 `hbm_runtime` 和 CPU 文本 encoder 的 `onnxruntime`。未固定特定镜像或固件版本。
- 板端推理还需 `onnxruntime`、两个模型制品和内置 BPE 词表。

<a id="quickstart"></a>
## 快速体验

显式准备两个模型，再从仓库根目录在 X5 上运行。`run.sh` 是快捷入口，不会自动下载模型。

```bash
# cwd：仓库根目录；来源：docs/release/x5/models.yaml 中的精确 URL
python3 samples/vision/clip/model/download.py --target x5
# 预期：samples/vision/clip/model/img_encoder.bin 和 text_encoder.onnx

# cwd：仓库根目录；输入：test_data/dog.jpg；prompt：默认 a diagram,a dog
python3 samples/vision/clip/runtime/python/main.py --target x5
# 预期：打印 JSON prompts/scores/order，并生成 samples/vision/clip/test_data/inference.png；退出码 0
```

默认可视化路径是受版本控制的 `test_data/inference.png`，每次运行都会覆盖。传入 `--img-save-path` 可写到其他位置。`run.sh` 只是快捷入口，不准备模型。

<a id="expected-results"></a>
## 预期结果

CLI 打印 `target`、`prompts`、`scores`、`order`、`image_saved`。`scores` 按 prompt 原顺序保存 cosine similarity，`order` 是降序 prompt 索引。绘图会将每个 prompt 和分数写入输入图片的副本。`dog.jpg` 的定性预期是 `a dog` 的分数高于 `a diagram`（源验证预期）；没有公开数值 benchmark。

编码器生成 512 维特征。BPE 词表位于 `runtime/python/bpe_simple_vocab_16e6.txt.gz`；余弦分数保持 prompt 顺序，运行时同时返回降序排名。

<a id="entry-points"></a>
## 入口索引

- 模型准备：[`model/README_cn.md`](model/README_cn.md) —— X5 图像 `.bin`、文本 `.onnx`，分别使用独立 asset identity。
- Python 运行：[`runtime/python/README_cn.md`](runtime/python/README_cn.md) —— BPE tokenizer、BPU 图像 + CPU ONNX 文本推理、cosine 排序和绘图。
- C++ 运行：未提供；C++ 为 `not-supported`。
- 模型转换：[`conversion/README_cn.md`](conversion/README_cn.md) —— 源协议和缺失的导出/校准配方。
- 模型评估：[`evaluator/README_cn.md`](evaluator/README_cn.md) —— 验证命令与定性预期；没有公开 benchmark 数值。

<a id="license"></a>
## 许可说明

示例代码遵循仓库 [LICENSE](../../../LICENSE) 的 Apache-2.0。CLIP 制品的许可证和来源以 X5 发布记录为准。X5 源中的贡献者信息予以保留。
