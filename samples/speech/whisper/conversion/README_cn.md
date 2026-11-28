# Whisper 模型转换与编译指南

[English](./README.md) | 简体中文

本目录提供了将 Whisper 模型转换为适配地瓜机器人（D-Robotics）`RDK X5` 硬件的 BPU 量化模型（`.bin`）的工具与说明。

## 目录结构

```text
.
|-- onnx_export
|   |-- export_whisper_bpu.py
|-- generate_calibration_data.py
|-- whisper_encoder_tiny.yaml
|-- whisper_decoder_tiny.yaml   （解码器不部署到 BPU，仅供参考）
|-- README.md
`-- README_cn.md
```

## 模型编译环境

编译使用配套工程 `RDK_x5_asr` 提供的 `x5-asr-compile` 和 OpenExplorer X5 CPU 1.2.8。环境准备见该工程的 `README.md`。

从 PyTorch 导出时，请在 PC 上准备 `torch`、`openai-whisper`、`onnx` 和 `onnxsim`。已有 ONNX 和校准张量时可直接执行第 3 步。

## 转换流程

### 1. PyTorch → ONNX

将 Whisper 编码器和解码器导出为 ONNX 格式：

```bash
cd samples/speech/whisper/conversion
python3 onnx_export/export_whisper_bpu.py --model_type tiny
```

导出器将编码器、支持变长 token 的 `whisper_decoder_tiny.onnx` 和 `multilingual.tiktoken` 写入 `../model/`。离线导出可传 `--checkpoint /path/to/tiny.pt`。以下配置对应多语言 Tiny。

### 2. 准备校准数据

``hb_mapper`` 使用预计算的 log-Mel 频谱校准，建议准备 20–50 条有代表性的语音。
使用提供的脚本从 WAV 音频生成：

```bash
# 从 WAV 音频目录生成校准数据
python3 generate_calibration_data.py \
    --audio-dir /path/to/wav/files \
    --mel-filters ../model/mel_80_filters.txt \
    --output-dir ../cal_data \
    --num-samples 50
```

脚本在 `../cal_data/` 中为每段音频写出一个无头 float32 `.bin` 文件。每份张量的形状为 `(1, 80, 3000, 1)`，大小为 960000 字节。编译命令也可读取已有的 float32 `.npy` 校准张量。

**校准数据要求：**
- 输入：WAV 音频（任意采样率，会自动重采样到 16 kHz）
- 20–50 个覆盖典型语音模式的多样化音频样本

### 3. 编译 X5 模型

在 ASR 转换环境中进入 `samples/speech/whisper/conversion/`，执行以下命令。

```bash
x5-asr-compile whisper_encoder_tiny.yaml
cp whisper_encoder_tiny_80x3000_x5/whisper_encoder_tiny_80x3000_x5.bin ../model/
```

编译结果是 X5 混合精度编码器。板端通过 ONNX Runtime CPU 运行 `whisper_decoder_tiny.onnx`。

---

## 支持的变种

| 变种 | 编码器 ONNX | 输出 `.bin` |
| --- | --- | --- |
| Tiny | `whisper_encoder_tiny.onnx` | `whisper_encoder_tiny_80x3000_x5.bin` |

---

## 常见问题

### "file does not start with RIFF id"

parquet 中的音频 bytes 不是 WAV 格式（没有 RIFF 头），通常是 **MP3** 格式
（以 ``ID3`` 开头）或其他压缩格式。

**解决**：安装 ``soundfile`` 作为通用解码后端：

```bash
pip install soundfile
```

校准脚本会自动为 WAV/FLAC 以外的格式回退到 soundfile（支持 MP3、OGG 等）。

### "No module named 'scipy'"

非 16 kHz 采样率的音频需要重采样，依赖 ``scipy``。

**解决**：

```bash
pip install scipy
```

### "Row N: unsupported audio column type dict, keys=['bytes', 'path']"

parquet 的 ``audio`` 列缺少 ``sampling_rate`` 键。脚本现在会通过 soundfile
自动从音频数据中提取采样率，只要安装了 soundfile，此警告可忽略。

### WARNING: Fewer than 20 calibration samples

校准样本少于 20 个时 INT8 量化精度可能欠佳。建议使用 ``--num-samples 50``
或更多来保证量化质量。

### 校准脚本的 Python 依赖（PC 端）

校准脚本在 **PC 端**（非板端）运行，需要：

```bash
pip install numpy pandas pyarrow soundfile scipy
```

---

## 输出协议

当前 runtime sample 默认协议如下：

- 编码器输入为 float32 log-Mel featuremap，形状为 `(1, 80, 3000, 1)`。
- 编码器输出为形状 `(1, 1500, 384)` 的隐藏状态，供解码器生成 token。

---

## License
本目录下的工具遵循 [Apache 2.0 License](../../../../LICENSE)。
