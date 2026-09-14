# Model Files

[简体中文](./README_cn.md) | English

This directory provides the prebuilt HGNetV2 deployment models for `RDK X5`. The current runtime path uses `.bin` models with `hbm_runtime`.

## Directory Structure

```text
.
├── download.sh
├── README.md
└── README_cn.md
```

## Available Models

| Variant | File Name | Size |
| --- | --- | --- |
| HGNetV2 b0 | `hgnetv2_b0_224x224_nv12.bin` | ~5.9 MB |
| HGNetV2 b1 | `hgnetv2_b1_224x224_nv12.bin` | ~6.2 MB |
| HGNetV2 b2 | `hgnetv2_b2_224x224_nv12.bin` | ~11 MB |
| HGNetV2 b3 | `hgnetv2_b3_224x224_nv12.bin` | ~16 MB |
| HGNetV2 b4 | `hgnetv2_b4_224x224_nv12.bin` | ~19 MB |

The default model used by `runtime/python/run.sh` and `runtime/python/main.py` is `hgnetv2_b0_224x224_nv12.bin`.

## Download Models

By default the script downloads only the b0 variant used by `runtime/python/run.sh`:

```bash
chmod +x download.sh
./download.sh                 # b0 only (~5.9 MB)
```

Pass variant names (or `all`) to fetch the others:

```bash
./download.sh b3 b4           # download b3 and b4
./download.sh all             # download all five variants (~57 MB)
```

The script downloads the `.bin` files into this directory and skips any that are already present.
