# Model Files

This directory provides prebuilt deployment models for the FastViT sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Model List

- `FastViT_SA12_224x224_nv12.bin`
- `FastViT_S12_224x224_nv12.bin`
- `FastViT_T12_224x224_nv12.bin`
- `FastViT_T8_224x224_nv12.bin`

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `FastViT_S12_224x224_nv12.bin`

## Download Models

Run the following script to download all FastViT models:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model files into this directory.
