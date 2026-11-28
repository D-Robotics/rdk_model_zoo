# Whisper Tiny model files

English | [简体中文](./README_cn.md)

Follow the [conversion guide](../conversion/README.md) to export Tiny and compile the X5 encoder. Place the resulting files in this directory.

| File | Purpose |
| --- | --- |
| `whisper_encoder_tiny_80x3000_x5.bin` | Mixed-precision X5 encoder with float32 log-Mel input |
| `whisper_decoder_tiny.onnx` | ONNX Runtime CPU decoder with a variable-length token sequence |
| `multilingual.tiktoken` | Multilingual vocabulary for the same checkpoint |

The runtime reads the sample's `mel_80_filters.txt`. An existing decoder `.bin` and its vocabulary can also be selected using the runtime arguments.
