# Wav2Vec2 Base 960h model files

English | [简体中文](./README_cn.md)

Follow the [conversion guide](../conversion/README.md) to export and compile `facebook/wav2vec2-base-960h`. Place `wav2vec2_base_960h_1x65535_x5.bin` in this directory alongside the included English CTC `vocab.json`. The default input is 65,535 mono samples at 16 kHz.

Use `--model-path` to select an existing board model and `--vocab-path` to select the matching vocabulary for a different checkpoint.
