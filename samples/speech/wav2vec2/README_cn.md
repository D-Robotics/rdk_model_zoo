# Wav2Vec2 模型说明

[English](./README.md) | 简体中文

本目录给出 Wav2Vec2 sample 在 Model Zoo 中的完整使用说明，包括算法概览、模型转换、运行时推理、模型文件管理和评测说明。

## 算法介绍

Wav2Vec2 是 Meta AI 开发的自监督语音表示学习模型，通过卷积特征编码器和基于 Transformer 的上下文网络从原始音频中学习上下文化的语音表示，并通过标注数据结合 CTC 进行微调用于自动语音识别（ASR）。

- **论文**: [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
- **官方实现**: [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)

### 算法功能

Wav2Vec2 支持以下任务：

- 自动语音识别（ASR）转录
- 音素识别
- 语音特征提取

### 算法特点

- **自监督预训练**：通过对比学习从未标注音频数据中学习语音表示。
- **卷积编码器**：使用时域卷积将原始音频波形映射为潜在特征序列。
- **Transformer 上下文网络**：在潜在特征上构建上下文化表示。
- **CTC 解码**：使用连接时序分类（CTC）损失进行序列到序列 ASR 微调。

## 目录结构

```text
conversion/       # 导出 ONNX、准备校准数据、编译 X5 模型
evaluator/        # 转录精度评估
model/            # 部署模型、词表和获取说明
runtime/python/   # Python 推理入口和运行脚本
test_data/        # 示例 WAV 与对应转录文本
```

## 快速体验

### Python

先按[模型说明](./model/README_cn.md)准备部署文件。在 RDK X5 上执行，脚本会选取本地或板端已安装的模型，并读取示例自带的 WAV。

- Python 详细说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。
- 快速体验命令：

```bash
cd runtime/python
bash run.sh
```

## 模型转换

- 部署文件及其准备方法见[模型说明](./model/README_cn.md)。
- 转换说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 模型推理

本 sample 当前维护的推理路径为 Python。

- Python 推理说明: [runtime/python/README_cn.md](./runtime/python/README_cn.md)

## 模型评估

评测说明、性能数据和验证结果请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 推理结果

自带的 `test_data/test_audio.wav` 在 X5 工具链量化仿真中的转录为 **THE TRAIN IS COMING**。实板 WER、延迟和实时率尚待测量。

## License

遵循 Model Zoo 顶层 License。
