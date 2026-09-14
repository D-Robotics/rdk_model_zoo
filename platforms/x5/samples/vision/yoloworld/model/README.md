# Model Download

This directory manages the YOLOWorld model file used by the Python sample.

## Default Model

| Model | Input | Format | Runtime |
| --- | --- | --- | --- |
| `yolo_world.bin` | image + text embeddings | `.bin` | `hbm_runtime` |

## Download

```bash
bash download.sh
```

The script downloads the model from the RDK X5 Model Zoo archive when it is not already present in this directory.
