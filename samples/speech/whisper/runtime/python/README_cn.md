# Whisper ASR Python 示例

[English](./README.md) | 简体中文

板端通过 BSP 的 `hbm_runtime` 加载 X5 编码器，通过 `onnxruntime` 的 CPU 后端运行 Whisper Tiny 解码器。音频预处理需要 NumPy，PCM WAV 读取可用 SciPy 或 SoundFile。

## 目录结构

```text
.
|-- main.py
|-- whisper.py
|-- README.md
|-- README_cn.md
`-- run.sh
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--encoder-model-path` | BPU 编码器 `.bin` 模型路径。 | `../../model/whisper_encoder_tiny_80x3000_x5.bin` |
| `--decoder-model-path` | CPU 解码器 ONNX 路径，也可指定已有的解码器 `.bin`。 | `../../model/whisper_decoder_tiny.onnx` |
| `--vocab-path` | 词汇表文本文件路径。 | `../../model/multilingual.tiktoken` |
| `--mel-filters-path` | Mel 滤波器组数据文件路径。 | `../../model/mel_80_filters.txt` |
| `--priority` | 模型优先级，范围 `0~255`。 | `0` |
| `--bpu-cores` | 用于推理的 BPU 核索引。 | `0` |
| `--test-audio` | 测试输入音频文件路径（WAV）。 | `../../test_data/test_audio.wav` |
| `--text-save-path` | 转录文本保存路径。 | `../../test_data/result.txt` |
| `--language` | 转录语言代码（如 en, zh）。 | `en` |

## 快速运行

```bash
chmod +x run.sh
./run.sh
```

## 手动运行

- 使用默认参数：

```bash
python3 main.py
```

- 显式指定参数：

```bash
python3 main.py \
    --encoder-model-path ../../model/whisper_encoder_tiny_80x3000_x5.bin \
    --decoder-model-path ../../model/whisper_decoder_tiny.onnx \
    --test-audio ../../test_data/test_audio.wav \
    --language en
```

## 接口说明

- **WhisperConfig**：封装编码器/解码器模型路径、词汇表和 Mel 滤波器组路径。
- **Whisper**：实现音频预处理（log-Mel 频谱）、X5 编码器推理和 CPU 自回归解码。
