# Model Download

This directory manages the CLIP model files used by the Python sample.

## Default Models

| Model | Role | Format | Runtime |
| --- | --- | --- | --- |
| `img_encoder.bin` | Image encoder | `.bin` | `hbm_runtime` |
| `text_encoder.onnx` | Text encoder | `.onnx` | `onnxruntime` |

## Download

```bash
bash download.sh
```

The script downloads both model files from the RDK X5 Model Zoo archive when they are not already present in this directory.
