# Model Files

This directory provides the prebuilt deployment model for the PP-LiteSeg-STDC1 sample on `RDK X5`.

The current runtime path uses `.bin` models with `hbm_runtime`.

## Directory Structure

```text
.
├── download.sh
├── README.md
└── README_cn.md
```

## Default Model

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is:

- `pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin`

## Download Model

Run the following script to download the default model:

```bash
chmod +x download.sh
./download.sh
```

The script downloads the `.bin` model file into this directory.

> To build the model yourself, follow [../conversion/README.md](../conversion/README.md).
