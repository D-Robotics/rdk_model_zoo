English | [简体中文](README_cn.md)

# MobileOne model artifacts

<a id="artifacts"></a>
## Artifacts

This directory lists the X5 release assets for the single-stage classifier. Choose a target and variant from the sample support matrix.

| Variant | Filename | Target | Format | Source |
| --- | --- | --- | --- | --- |
| `s0` | `MobileOne_S0_224x224_nv12.bin` | x5 | bin | download |
| `s1` | `MobileOne_S1_224x224_nv12.bin` | x5 | bin | download |
| `s2` | `MobileOne_S2_224x224_nv12.bin` | x5 | bin | download |
| `s3` | `MobileOne_S3_224x224_nv12.bin` | x5 | bin | download |
| `s4` | `MobileOne_S4_224x224_nv12.bin` | x5 | bin | download |

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

cwd: repository root. Shell arguments are positional; the Python form uses named options. Omitting a variant selects `s0` in both forms.

```bash
bash samples/vision/mobileone/model/download.sh x5 s0
python3 samples/vision/mobileone/model/download.py --target x5 --variant s4
```

Output: exit 0, filename and observed SHA-256 printed. Download writes a temporary file then installs without overwriting existing bytes. Size/hash failure exits with an error. If the server is unavailable, transfer the matching published file manually into `model/`; preserve its source and observed digest. Do not substitute a different variant.

<a id="accompanying-files"></a>
## Accompanying files

The CLI reads `datasets/imagenet/imagenet_classes.names` for labels and `test_data/tiger_beetle.JPEG` as its default image. The API accepts an in-memory image and optional labels; without labels it returns class IDs as strings. No tokenizer or extra model is required.

<a id="local-paths"></a>
## Local paths

Files land at `samples/vision/mobileone/model/<filename>`. The default runtime selection resolves `MobileOne_S0_224x224_nv12.bin` there. An external `--model-path` requires its exact `--asset-id`, e.g. `x5:mobileone:MobileOne_S0_224x224_nv12.bin`.

<a id="formats-checksums"></a>
## Formats and checksums

For every file above: `format: bin`, `sha256: null (unknown)` in `docs/release/x5/models.yaml`. The downloader prints the computed digest after transfer; the manifest SHA-256 is `null (unknown)`.

<!-- converted-batch-20261102 -->
## Converted-only additions (2026-11-02)

These independently converted timm weights have passed strict checkpoint loading,
reparameterization and two-input PyTorch/ONNX host comparison, checker and compiler.
Board smoke, accuracy, Runtime 1/2 and C++ 1/2 are **not-run**; historical metrics
and screenshots do not describe these artifacts. Existing defaults are retained.
Pillow is needed for the explicit bicubic shorter-edge 256 / rounded center crop
224 profile. Normalization is fused in Mapper from the actual checkpoint config;
output is float32 1,000-class logits followed by native softmax.

| Variant | Artifact | Checkpoint revision | Board |
|---|---|---|---|
| `timm-s0-apple-in1k` | `mobileone_s0_apple_in1k_bayese_224x224_nv12.bin` | [`2f1da1a4433b`](https://huggingface.co/timm/mobileone_s0.apple_in1k/blob/2f1da1a4433b0636bace873cab611c9ee90d40de/README.md) | not-run |
| `timm-s1-apple-in1k` | `mobileone_s1_apple_in1k_bayese_224x224_nv12.bin` | [`c4a13fe645a4`](https://huggingface.co/timm/mobileone_s1.apple_in1k/blob/c4a13fe645a42cbb683439dfb1934c815155f2a0/README.md) | not-run |
| `timm-s2-apple-in1k` | `mobileone_s2_apple_in1k_bayese_224x224_nv12.bin` | [`0cd6548c3117`](https://huggingface.co/timm/mobileone_s2.apple_in1k/blob/0cd6548c311753d2dd525d414c22fce35738264c/README.md) | not-run |
| `timm-s3-apple-in1k` | `mobileone_s3_apple_in1k_bayese_224x224_nv12.bin` | [`29a687f5d735`](https://huggingface.co/timm/mobileone_s3.apple_in1k/blob/29a687f5d735adb9ee23667dce14b60e8eaa6e88/README.md) | not-run |

```bash
python3 samples/vision/mobileone/model/download.py --target x5 --variant timm-s0-apple-in1k
python3 samples/vision/mobileone/runtime/python/main.py --target x5 --variant timm-s0-apple-in1k --dry-run
```

Exact model URLs/SHA-256/size are in `docs/release/x5/models.yaml`; source and host
receipts are in `conversion/converted.json`. Each weight retains its own model
license in its public release manifest. No board result is implied by download.
<!-- /converted-batch-20261102 -->
