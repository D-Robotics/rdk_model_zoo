# Model Files

This directory provides prebuilt deployment models for the RepViT sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Model List

- `RepViT_m0_9_224x224_nv12.bin`
- `RepViT_m1_0_224x224_nv12.bin`
- `RepViT_m1_1_224x224_nv12.bin`

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `RepViT_m0_9_224x224_nv12.bin`

## Download Models

Run the following script to download all RepViT models:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model files into this directory.
