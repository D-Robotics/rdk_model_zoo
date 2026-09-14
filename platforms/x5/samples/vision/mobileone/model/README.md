# Model Files

This directory provides the prebuilt deployment models for the MobileOne sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Model List

- `MobileOne_S0_224x224_nv12.bin`
- `MobileOne_S1_224x224_nv12.bin`
- `MobileOne_S2_224x224_nv12.bin`
- `MobileOne_S3_224x224_nv12.bin`
- `MobileOne_S4_224x224_nv12.bin`

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `MobileOne_S0_224x224_nv12.bin`

## Download Model

Run the following script to download the MobileOne models:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model files into this directory.
