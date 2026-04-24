# Model Files

This directory provides prebuilt deployment models for the EdgeNeXt sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Model List

- `EdgeNeXt_base_224x224_nv12.bin`
- `EdgeNeXt_small_224x224_nv12.bin`
- `EdgeNeXt_x_small_224x224_nv12.bin`
- `EdgeNeXt_xx_small_224x224_nv12.bin`

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `EdgeNeXt_base_224x224_nv12.bin`

## Download Models

Run the following script to download all EdgeNeXt models:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model files into this directory.
