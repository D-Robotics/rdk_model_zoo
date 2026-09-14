# Model Files

This directory provides the prebuilt deployment models for the RepGhost sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Model List

- `RepGhost_100_224x224_nv12.bin`
- `RepGhost_111_224x224_nv12.bin`
- `RepGhost_130_224x224_nv12.bin`
- `RepGhost_150_224x224_nv12.bin`
- `RepGhost_200_224x224_nv12.bin`

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `RepGhost_100_224x224_nv12.bin`

## Download Model

Run the following script to download the RepGhost models:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model files into this directory.
