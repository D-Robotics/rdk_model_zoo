# Model Files

This directory provides the prebuilt deployment models for the RepVGG sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Model List

- `RepVGG_A0_224x224_nv12.bin`
- `RepVGG_A1_224x224_nv12.bin`
- `RepVGG_A2_224x224_nv12.bin`
- `RepVGG_B0_224x224_nv12.bin`
- `RepVGG_B1g2_224x224_nv12.bin`
- `RepVGG_B1g4_224x224_nv12.bin`

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `RepVGG_A0_224x224_nv12.bin`

## Download Model

Run the following script to download the RepVGG models:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model files into this directory.
