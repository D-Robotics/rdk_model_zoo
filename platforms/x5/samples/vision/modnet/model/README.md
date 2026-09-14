# Model Files

This directory provides the prebuilt deployment model for the MODNet sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Directory Structure

```text
.
├── download_model.sh
├── README.md
└── README_cn.md
```

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `modnet_512x512_rgb.bin`

## Download Model

Run the following script to download the default model:

```bash
chmod +x download_model.sh
./download_model.sh
```

The script downloads the `.bin` model file into this directory.

**Note**: The model download URL is not yet available. If the download script fails, please manually place the `modnet_512x512_rgb.bin` file in this directory.
