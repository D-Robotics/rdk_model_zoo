# Model Files

This directory provides the prebuilt deployment model for the GoogLeNet sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Model List

- `googlenet_224x224_nv12.bin`

## Default Model

The default model used by `runtime/python/main.py` is the board-side preinstalled model:

- `/opt/hobot/model/x5/basic/googlenet_224x224_nv12.bin`

The local model downloaded by `model/download.sh` is:

- `googlenet_224x224_nv12.bin`

## Download Models

Run the following script to download the model:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model file into this directory.
