English | [简体中文](./README_cn.md)

# MobileNetV2 Image Classification Sample (Python)

This sample demonstrates how to use the `HB_HBMRuntime` Python API to deploy a MobileNetV2 model for image classification, outputting Top-K class IDs and their confidence scores. Designed for RDK series devices with BPU.

## Dependencies

No special dependencies beyond:

```bash
pip install numpy==1.26.4 opencv-python==4.11.0.86 scipy==1.15.3
```

## Directory Structure

```text
.
├── README.md           # This file
├── main.py             # Entry script — runs MobileNetV2 classification
├── mobilenetv2.py      # MobileNetV2 wrapper (preprocessing, inference, post-processing)
└── run.sh              # One-click run script (env setup, model download, execution)
```

## Arguments

| Argument | Description | Default |
|---|---|---|
| `--model-path` | Path to the `.hbm` model file | `/opt/hobot/model/<soc>/basic/mobilenetv2_224x224_nv12.hbm` |
| `--test-img` | Path to the test image | `../../test_data/zebra_cls.jpg` |
| `--label-file` | Path to the label mapping file | `../../test_data/imagenet1000_labels.txt` |
| `--priority` | Model priority (0~255, higher = higher) | `0` |
| `--bpu-cores` | BPU core indexes (e.g. `--bpu-cores 0 1`) | `[0]` |

> **Note**: `<soc>` in the default `--model-path` is resolved at runtime based on the board (e.g. `s100`, `s600`).

## Quick Run

- Use the one-click script:
    ```bash
    ./run.sh
    ```

- Use defaults:
    ```bash
    python main.py
    ```

- Custom arguments:
    ```bash
    python main.py \
    --model-path /opt/hobot/model/s100/basic/mobilenetv2_224x224_nv12.hbm \
    --test-img ../../test_data/zebra_cls.jpg \
    --label-file ../../test_data/imagenet1000_labels.txt
    ```

## Expected Output

```bash
Top-5 Classification Results:
[0] zebra: 0.9922
[1] tiger, Panthera tigris: 0.0040
[2] hartebeest: 0.0013
[3] tiger cat: 0.0007
[4] impala, Aepyceros melampus: 0.0005
```

## API Reference

See the [source reference docs](../../../../../docs/source_reference/README.md) for detailed API documentation.