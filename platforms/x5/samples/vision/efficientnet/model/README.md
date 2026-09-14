# Model Files

This directory provides prebuilt deployment models for the EfficientNet sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Model List

- `EfficientNet_B2_224x224_nv12.bin`
- `EfficientNet_B3_224x224_nv12.bin`
- `EfficientNet_B4_224x224_nv12.bin`

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `EfficientNet_B2_224x224_nv12.bin`

## Download Models

Run the following script to download all EfficientNet models:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model files into this directory.
