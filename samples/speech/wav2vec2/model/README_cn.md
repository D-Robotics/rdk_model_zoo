# Wav2Vec2 Base 960h 模型文件

[English](./README.md) | 简体中文

按[转换步骤](../conversion/README_cn.md)从 `facebook/wav2vec2-base-960h` 导出并编译模型。将 `wav2vec2_base_960h_1x65535_x5.bin` 放入本目录，随附的 `vocab.json` 用于英文 CTC 解码。默认输入为 16 kHz、65,535 个单声道采样点。

已有匹配的板端模型也可通过 `--model-path` 指定。如使用其他检查点，请用 `--vocab-path` 传入对应的词表。
