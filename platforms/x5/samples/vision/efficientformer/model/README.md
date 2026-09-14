# Model Files

This directory provides prebuilt deployment models for the EfficientFormer sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Model List

- `EfficientFormer_l1_224x224_nv12.bin`
- `EfficientFormer_l3_224x224_nv12.bin`

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `EfficientFormer_l3_224x224_nv12.bin`

## Download Models

Run the following script to download all EfficientFormer models:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model files into this directory.
