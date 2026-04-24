# Model Files

This directory provides prebuilt deployment models for the EfficientFormerV2 sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Model List

- `EfficientFormerv2_s0_224x224_nv12.bin`
- `EfficientFormerv2_s1_224x224_nv12.bin`
- `EfficientFormerv2_s2_224x224_nv12.bin`

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `EfficientFormerv2_s0_224x224_nv12.bin`

## Download Models

Run the following script to download all EfficientFormerV2 models:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model files into this directory.
