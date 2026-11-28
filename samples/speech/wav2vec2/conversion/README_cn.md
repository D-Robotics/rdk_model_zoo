# Wav2Vec2 模型转换与编译指南

[English](./README.md) | 简体中文

本目录提供了将 Wav2Vec2 模型转换为适配地瓜机器人（D-Robotics）`RDK X5` 硬件的 BPU 量化模型（`.bin`）的工具与说明。

## 目录结构

```text
.
|-- onnx_export
    |-- export_wav2vec2_bpu.py
    |-- requirements.txt
|-- wav2vec2_base.yaml
|-- README.md
`-- README_cn.md
```

## 模型编译环境

编译使用配套工程 `RDK_x5_asr` 提供的 `x5-asr-compile` 和 OpenExplorer X5 CPU 1.2.8。环境准备见该工程的 `README.md`。

## 转换流程

### 1. 导出静态 ONNX

在 PC 的独立 Python 环境中，从仓库根目录执行以下命令。已有 ONNX 时可跳至校准步骤。

```bash
cd samples/speech/wav2vec2/conversion
python3 -m pip install -r onnx_export/requirements.txt
python3 onnx_export/export_wav2vec2_bpu.py
```

默认导出 `facebook/wav2vec2-base-960h` 到 `../model/wav2vec2_base.onnx`，输入是 `[1, 65535]` 的 16 kHz 波形。`--samples` 指定导出的固定长度。`--checkpoint` 接受本地权重目录或其他 CTC 检查点，例如 `ccoreilly/wav2vec2-large-100k-voxpopuli-catala`。

### 2. 准备校准波形

从当前 `conversion/` 目录运行，使用代表性的语音文件。脚本读取 ONNX 的实际输入长度，转为 16 kHz 单声道，补齐或截断后将 float32 张量写入 `../cal_data/`。每份默认校准张量占 262140 字节。

```bash
python3 generate_calibration_data.py --audio-dir /path/to/speech --num-samples 50
```

### 3. 编译 X5 模型

在 ASR 转换环境中进入 `samples/speech/wav2vec2/conversion/`，执行以下命令。

```bash
x5-asr-compile wav2vec2_base.yaml
cp wav2vec2_base_960h_1x65535_x5/wav2vec2_base_960h_1x65535_x5.bin ../model/
```

输入为 float32 波形，长度取自导出的 ONNX。更改固定长度时，请同步更新 YAML 的输出名称和运行参数。

对每条固定长度波形做零均值、单位方差归一化，校准、浮点对照和板端运行共用同一实现。示例的 `vocab.json` 来自 Base 960h 检查点。更换 CTC 词表时，将配套的 `vocab.json` 放在 `.bin` 旁，或使用 `--vocab-path`。

## 支持的变种

| 变种 | 模型 ID | 输出 `.bin` |
| --- | --- | --- |
| Base | `facebook/wav2vec2-base-960h` | `wav2vec2_base_960h_1x65535_x5.bin` |

---

## 输出协议

当前 runtime sample 默认协议如下：

- 输入张量：重采样至 16 kHz 的原始音频波形
- 输出张量：用于解码的 CTC 发射 logits

---

## 板端精度验证

编译完成后，建议使用 `runtime/python/compare_accuracy.py` 对比浮点 ONNX 与 BPU 量化模型的 CTC logits，量化损失以**余弦相似度**（建议 `>= 0.95`）、逐帧 argmax 一致率和 token 级 CER 衡量：

1. 在转换 PC 上导出浮点 logits 参考：
   ```bash
   cd ../runtime/python
   python3 compare_accuracy.py \
       --float-onnx ../../model/wav2vec2_base.onnx \
       --test-audio ../../test_data/test_audio.wav \
       --save-float-logits ../../test_data/float_logits.npz
   ```
2. 将 `float_logits.npz` 与 `.bin` 拷到板端后对比：
   ```bash
   python3 compare_accuracy.py \
       --float-logits ../../test_data/float_logits.npz \
       --bin-model ../../model/wav2vec2_base_960h_1x65535_x5.bin \
       --test-audio ../../test_data/test_audio.wav
   ```

若余弦相似度低于阈值，请检查校准数据是否具有代表性（20–50 条与部署场景相近的音频），或调整编译参数后重新编译。

---

## License
本目录下的工具遵循 [Apache 2.0 License](../../../../LICENSE)。
