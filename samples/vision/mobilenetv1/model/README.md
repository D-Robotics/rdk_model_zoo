# Model Files

This directory provides the prebuilt deployment model for the MobileNetV1 sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `mobilenetv1_224x224_nv12.bin`

## Download Model

Run the following script to download the default model:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model file into this directory.
