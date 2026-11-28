# Wav2Vec2 ASR Python 示例

[English](./README.md) | 简体中文

本示例展示如何在 BPU 上使用量化后的 Wav2Vec2 模型执行自动语音识别（ASR）任务，使用 CTC 解码。

## 目录结构

```text
.
|-- main.py
|-- compare_accuracy.py
|-- wav2vec2.py
|-- README.md
|-- README_cn.md
`-- run.sh
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-path` | 量化 `.bin` 模型文件路径。 | `../../model/wav2vec2_base_960h_1x65535_x5.bin` |
| `--priority` | 模型优先级，范围 `0~255`。 | `0` |
| `--bpu-cores` | 用于推理的 BPU 核索引。 | `0` |
| `--test-audio` | 测试输入音频文件路径（WAV）。 | `../../test_data/test_audio.wav` |
| `--text-save-path` | 转录文本保存路径。 | `../../test_data/result.txt` |
| `--language` | 转录语言代码（如 en, zh）。 | `en` |
| `--blank-id` | CTC 空白 token ID。 | `0` |
| `--vocab-path` | CTC 词表路径。 | 模型目录内的 `vocab.json` 或随样例提供的词表 |

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
    --model-path ../../model/wav2vec2_base_960h_1x65535_x5.bin \
    --test-audio ../../test_data/test_audio.wav \
    --language en \
    --blank-id 0
```

## 接口说明

- **Wav2Vec2Config**：封装模型路径、语言和 CTC 空白 token ID。
- **Wav2Vec2**：实现音频预处理、BPU 推理和 CTC 贪心解码到文本。

## 板端精度损失验证

`compare_accuracy.py` 通过对比浮点模型与 BPU 量化模型输出的 CTC logits，量化 INT8 量化带来的精度损失。指标包括：

- **余弦相似度（cosine similarity）**：工具链常用的量化损失指标，通常 `>= 0.95` 认为可接受；
- **逐帧 argmax 一致率**：直接预测贪心 CTC 解码结果是否变化，非空白帧一致率更有参考价值；
- **token 级 CER**：对比两条解码 token 序列的编辑距离；
- 数值误差统计（MAE / 最大绝对误差）。

### 第一步：在转换 PC 上导出浮点参考

转换完成得到 `wav2vec2_base.onnx` 后，在 PC 上运行一次，把浮点 logits 保存为 `.npz`（无需 BPU）：

```bash
python3 compare_accuracy.py \
    --float-onnx ../../model/wav2vec2_base.onnx \
    --test-audio ../../test_data/test_audio.wav \
    --save-float-logits ../../test_data/float_logits.npz
```

### 第二步：在板端对比

把 `float_logits.npz` 与编译产物 `.bin` 一起拷到板端后运行：

```bash
python3 compare_accuracy.py \
    --float-logits ../../test_data/float_logits.npz \
    --bin-model ../../model/wav2vec2_base_960h_1x65535_x5.bin \
    --test-audio ../../test_data/test_audio.wav
```

板端 CPU 性能足够时，也可以直接指定 `--float-onnx` 让 onnxruntime 在线计算浮点参考。

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--float-onnx` | 浮点 ONNX 模型路径（opset 11）。 | 无 |
| `--float-logits` | 预先导出的浮点 logits `.npz`。 | 无 |
| `--bin-model` | BPU 量化 `.bin` 模型路径，省略则只导出浮点 logits。 | 无 |
| `--test-audio` / `--audio-dir` | 单条测试音频 / 音频目录（目录模式下结果为均值）。 | `../../test_data/test_audio.wav` |
| `--save-float-logits` | 浮点 logits 保存路径。 | 无 |
| `--cos-threshold` | 余弦相似度 PASS 阈值。 | `0.95` |
| `--blank-id` | CTC 空白 token ID。 | `0` |
| `--vocab-path` | CTC 词表路径。 | 模型目录内的 `vocab.json` 或随样例提供的词表 |
| `--report-path` | JSON 报告保存路径。 | `../../test_data/wav2vec2_accuracy_report.json` |

运行结束会输出逐样本指标、多样本均值以及 `PASS/FAIL` 判定；当余弦相似度低于阈值时，建议检查校准数据是否具有代表性，或调整编译参数。
