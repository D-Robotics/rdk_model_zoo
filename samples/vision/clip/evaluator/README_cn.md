[English](README.md) | 简体中文

# 模型评估 — CLIP 图文匹配

本文记录 CLIP 制品对的验证路径。没有发布 benchmark 表，因此不提供延迟或精度数值。

<a id="dataset"></a>
## 数据集

默认验证输入为 `samples/vision/clip/test_data/dog.jpg`，prompt 为 `a diagram` 和 `a dog`。BPE 词表随 runtime 提供：`samples/vision/clip/runtime/python/bpe_simple_vocab_16e6.txt.gz`。没有发布更大的评估数据集或准备脚本。

```text
# cwd：仓库根目录
samples/vision/clip/test_data/dog.jpg
samples/vision/clip/runtime/python/bpe_simple_vocab_16e6.txt.gz
# prompts：a diagram,a dog
```

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="environment"></a>
## 环境

使用 RDK X5 板端 `hbm_runtime` 执行图像编码器，以 CPU `onnxruntime` 执行文本编码器。Python 依赖为 NumPy、OpenCV、`ftfy==6.3.1` 和 `regex==2026.9.10`，词表随 runtime 提供。

<a id="command"></a>
## 评估命令

验证入口是 `runtime/python/main.py` 和 `run.sh`。下面的显式命令运行本 sample 并写入用户选择的图片：

```bash
# cwd：仓库根目录；前置：X5 两个模型制品已准备
python3 samples/vision/clip/runtime/python/main.py \
  --target x5 \
  --test-img samples/vision/clip/test_data/dog.jpg \
  --texts "a diagram,a dog" \
  --img-save-path /tmp/clip-eval/inference.png
# 预期：JSON scores/order 和 /tmp/clip-eval/inference.png；没有发布数值 benchmark
```

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| Cosine similarity | 每个文本特征与图像特征点积除以两者 L2 范数及 `1e-12` 稳定项。 | 一张图、N 个 prompt、float32 特征；分数保持 prompt 顺序。 |
| Rank order | 对 cosine 分数执行降序 `argsort`。 | 使用当前 NumPy 版本的 argsort 平局排序规则。 |

本 sample 没有发布延迟、top-k、检索或分类指标。

<a id="outputs"></a>
## 输出

runtime 将标注图片按 `--img-save-path` 精确写出，并打印 JSON `scores`/`order`。

<a id="reference-results"></a>
## 参考结果

没有公开数值 benchmark 表。验证预期是定性的：对于 `dog.jpg`，`a dog` 的分数应高于 `a diagram`。

| 参考项 | 值 | 条件 | 来源 |
| --- | --- | --- | --- |
| Dog prompt 排序 | `a dog` 排在 `a diagram` 之前 | X5 制品对、随附 BPE、`dog.jpg`、cosine 排序 | X5 平台源 evaluator README |

<a id="boundaries"></a>
## 适用范围

- 没有独立数据集评估器或公开 benchmark。
- 文本 encoder 是 CPU ONNX，板端需要 ONNX Runtime。
- 本 sample 只覆盖图文相似度，不包含 C++、训练或文本生成任务。

## 许可

评估文档遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。
