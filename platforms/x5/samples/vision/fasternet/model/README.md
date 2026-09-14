# Model Files

This directory provides prebuilt deployment models for the FasterNet sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Model List

- `FasterNet_S_224x224_nv12.bin`
- `FasterNet_T0_224x224_nv12.bin`
- `FasterNet_T1_224x224_nv12.bin`
- `FasterNet_T2_224x224_nv12.bin`

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `FasterNet_S_224x224_nv12.bin`

## Download Models

Run the following script to download all FasterNet models:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model files into this directory.
