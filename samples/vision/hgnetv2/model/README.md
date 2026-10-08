English | [简体中文](README_cn.md)

# HGNetV2 model artifacts

<a id="artifacts"></a>
## Artifacts

This directory lists the X5 release assets for the single-stage classifier. Choose a target and variant from the sample support matrix.

| Variant | Filename | Target | Format | Source |
| --- | --- | --- | --- | --- |
| `b0` | `hgnetv2_b0_224x224_nv12.bin` | x5 | bin | download |
| `b1` | `hgnetv2_b1_224x224_nv12.bin` | x5 | bin | download |
| `b2` | `hgnetv2_b2_224x224_nv12.bin` | x5 | bin | download |
| `b3` | `hgnetv2_b3_224x224_nv12.bin` | x5 | bin | download |
| `b4` | `hgnetv2_b4_224x224_nv12.bin` | x5 | bin | download |

### X5 file sizes

The X5 release model table (`rdk_x5 @ e3f9fa3fb5a795b2531bdb84fa60d03768af5956`)
lists these approximate BIN sizes in MB. Download validation uses the exact
byte count in `docs/release/x5/models.yaml`.

| Variant | File Name | Size |
| --- | --- | --- |
| HGNetV2 b0 | `hgnetv2_b0_224x224_nv12.bin` | ~5.9 MB |
| HGNetV2 b1 | `hgnetv2_b1_224x224_nv12.bin` | ~6.2 MB |
| HGNetV2 b2 | `hgnetv2_b2_224x224_nv12.bin` | ~11 MB |
| HGNetV2 b3 | `hgnetv2_b3_224x224_nv12.bin` | ~16 MB |
| HGNetV2 b4 | `hgnetv2_b4_224x224_nv12.bin` | ~19 MB |

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── __init__.py  # Python script
├── download.py  # Prepare model files
└── download.sh  # Model preparation command
```

<a id="preparation"></a>
## Preparation

cwd: repository root. Shell arguments are positional; the Python form uses named options. Omitting a variant selects `b0` in both forms.

```bash
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/model/download.py --target x5 --variant b4
```

Success: exit 0, filename and observed SHA-256 printed. Download writes a temporary file then installs without overwriting existing bytes. Size/hash failure exits with an error. If the server is unavailable, transfer the matching published file manually into `model/`; preserve its source and observed digest. Do not substitute a different variant.

<a id="accompanying-files"></a>
## Accompanying files

The CLI reads `datasets/imagenet/imagenet_classes.names` for labels and `test_data/sandbar.JPEG` as its default image. The API accepts an in-memory image and optional labels; without labels it returns class IDs as strings. No tokenizer or extra model is required.

<a id="local-paths"></a>
## Local paths

Files land at `samples/vision/hgnetv2/model/<filename>`. The default runtime selection resolves `hgnetv2_b0_224x224_nv12.bin` there. An external `--model-path` requires its exact `--asset-id`, e.g. `x5:hgnetv2:hgnetv2_b0_224x224_nv12.bin`.

<a id="formats-checksums"></a>
## Formats and checksums

For every file above: `format: bin`, `sha256: null (unknown)` in `docs/release/x5/models.yaml`. The downloader prints the computed digest after transfer; the manifest SHA-256 is `null (unknown)`.
