# Whisper 模型说明

[English](./README.md) | 简体中文

本目录给出 Whisper sample 在 Model Zoo 中的完整使用说明，包括算法概览、模型转换、运行时推理、模型文件管理和评测说明。

## 算法介绍

Whisper 是 OpenAI 开发的通用自动语音识别（ASR）模型，基于大规模多语言、多任务数据集训练，通过单一编码器-解码器 Transformer 架构同时支持转录、翻译和语言识别。

- **论文**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **官方实现**: [openai/whisper](https://github.com/openai/whisper)

### 算法功能

Whisper 支持以下任务：

- 多语言语音转文字转录
- 语音翻译（翻译为英文）
- 语言识别

### 算法特点

- **编码器-解码器 Transformer**：统一架构支持转录、翻译和语言检测。
- **多语言支持**：基于 68 万小时多语言语音数据训练，覆盖 99 种语言。
- **Log-Mel 频谱输入**：音频被转换为 80 通道 log-Mel 频谱图作为模型输入。
- **Token 输出**：模型通过多语言分词器输出文本 token。

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

- 板端需要编码器 `.bin`、CPU 解码器 ONNX 和词表，文件清单见 [model](./model/README_cn.md)。
- 转换说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 模型推理

本 sample 当前维护的推理路径为 Python。

- Python 推理说明: [runtime/python/README_cn.md](./runtime/python/README_cn.md)

## 模型评估

评测说明、性能数据和验证结果请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 推理结果

自带的 `test_data/test_audio.wav` 在 X5 工具链量化仿真中的转录为 **Please open the window.**。实板 WER、延迟和实时率尚待测量。

## License

遵循 Model Zoo 顶层 License。
