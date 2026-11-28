# Whisper Tiny 模型文件

[English](./README.md) | 简体中文

按[转换步骤](../conversion/README_cn.md)导出 Tiny 权重并编译 X5 编码器，然后将生成的文件放在本目录。

| 文件 | 用途 |
| --- | --- |
| `whisper_encoder_tiny_80x3000_x5.bin` | X5 混合精度编码器，输入为 float32 log-Mel |
| `whisper_decoder_tiny.onnx` | ONNX Runtime CPU 解码器，支持变长 token 序列 |
| `multilingual.tiktoken` | 同一检查点的多语言词表 |

运行时读取示例自带的 `mel_80_filters.txt`。已有的解码器 `.bin` 和配套词表也可通过运行参数指定。
