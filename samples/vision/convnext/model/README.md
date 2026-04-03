# Model Files

This directory contains the compiled BPU model files and scripts to download them.

## Directory Structure

```text
.
├── download.sh            # Script to download HBM models
├── README.md              # Documentation (English)
└── README_cn.md           # Documentation (Chinese)
```

## Download Models

To download the pre-compiled ConvNeXt models for RDK X5, run:

```bash
chmod +x download.sh
./download.sh
```

The script will download the `.bin` files into this directory.
