# Model Files

This directory provides the prebuilt deployment model for the EfficientViT sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Model List

- `EfficientViT_m5_224x224_nv12.bin`

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `EfficientViT_m5_224x224_nv12.bin`

## Download Model

Run the following script to download the EfficientViT model:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model file into this directory.
