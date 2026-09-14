# Model Files

This directory provides the prebuilt deployment models for the MobileNetV4 sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Model List

- `MobileNetV4_conv_small_224x224_nv12.bin`
- `MobileNetV4_conv_medium_224x224_nv12.bin`

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `MobileNetV4_conv_small_224x224_nv12.bin`

## Download Models

Run the following script to download both models:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model files into this directory.
