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

Output: exit 0, filename and observed SHA-256 printed. Download writes a temporary file then installs without overwriting existing bytes. Size/hash failure exits with an error. If the server is unavailable, transfer the matching published file manually into `model/`; preserve its source and observed digest. Do not substitute a different variant.

<a id="accompanying-files"></a>
## Accompanying files

The CLI reads `datasets/imagenet/imagenet_classes.names` for labels and `test_data/sandbar.JPEG` as its default image. The API accepts an in-memory image and optional labels; without labels it returns class IDs as strings. No tokenizer or extra model is required.

<a id="local-paths"></a>
## Local paths

Files land at `samples/vision/hgnetv2/model/<filename>`. The default runtime selection resolves `hgnetv2_b0_224x224_nv12.bin` there. An external `--model-path` requires its exact `--asset-id`, e.g. `x5:hgnetv2:hgnetv2_b0_224x224_nv12.bin`.

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
| `timm-b0-ssld-stage2-ft-in1k` | `hgnetv2_b0_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`9e0a904b210d`](https://huggingface.co/timm/hgnetv2_b0.ssld_stage2_ft_in1k/blob/9e0a904b210d65e4fcd6582f8ca6e0921fac2650/README.md) | not-run |
| `timm-b1-ssld-stage2-ft-in1k` | `hgnetv2_b1_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`ec7014d7f1d0`](https://huggingface.co/timm/hgnetv2_b1.ssld_stage2_ft_in1k/blob/ec7014d7f1d0dd1ea2bf3012bc51dc95c5919fd2/README.md) | not-run |
| `timm-b2-ssld-stage2-ft-in1k` | `hgnetv2_b2_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`7a3092bc8711`](https://huggingface.co/timm/hgnetv2_b2.ssld_stage2_ft_in1k/blob/7a3092bc87118c7974bb9000b1eb6d9c560f8f75/README.md) | not-run |
| `timm-b3-ssld-stage2-ft-in1k` | `hgnetv2_b3_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`1aac877b3ea5`](https://huggingface.co/timm/hgnetv2_b3.ssld_stage2_ft_in1k/blob/1aac877b3ea55ab0f06d57cad9889d1e601bd472/README.md) | not-run |
| `timm-b4-ssld-stage2-ft-in1k` | `hgnetv2_b4_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`3c808d9d9d98`](https://huggingface.co/timm/hgnetv2_b4.ssld_stage2_ft_in1k/blob/3c808d9d9d9873ed5e37b00509ba2a4882e816e8/README.md) | not-run |

```bash
python3 samples/vision/hgnetv2/model/download.py --target x5 --variant timm-b0-ssld-stage2-ft-in1k
python3 samples/vision/hgnetv2/runtime/python/main.py --target x5 --variant timm-b0-ssld-stage2-ft-in1k --dry-run
```

Exact model URLs/SHA-256/size are in `docs/release/x5/models.yaml`; source and host
receipts are in `conversion/converted.json`. Each weight retains its own model
license in its public release manifest. No board result is implied by download.
<!-- /converted-batch-20261102 -->

<!-- converted-all-20261103 -->

## Converted S profiles (2026-11-03)

These independent profiles passed conversion, complete OE collection and actual HBM metadata binding. Source weights, graphs, per-target recipes, calibration counts and digests are in [converted_targets_20261103.json](../conversion/converted_targets_20261103.json).

| Target | Native variant | Components | Calibration photographs |
| --- | --- | --- | ---: |
| s100 | `timm-b0-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-b0-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-b0-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-b1-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-b1-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-b1-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-b2-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-b2-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-b2-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-b3-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-b3-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-b3-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-b4-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-b4-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-b4-ssld-stage2-ft-in1k-s-int16-o1-20261103` | model | 200 |

```bash
python3 samples/vision/hgnetv2/model/download.py --target s100 --variant timm-b0-ssld-stage2-ft-in1k-s-int16-o1-20261103
python3 samples/vision/hgnetv2/runtime/python/main.py --target s100 --variant timm-b0-ssld-stage2-ft-in1k-s-int16-o1-20261103 --asset-id s:hgnetv2:s100/hgnetv2_b0_ssld_stage2_ft_in1k_nashe_224x224_nv12_float_int16_20261103_o1.hbm --model-path /path/to/hgnetv2_b0_ssld_stage2_ft_in1k_nashe_224x224_nv12_float_int16_20261103_o1.hbm --test-img samples/vision/hgnetv2/test_data/sandbar.JPEG
```

Substitute the actual file paths. Multi-component tasks require every component together. Historical defaults and measurements retain their original identities. Board smoke, dataset accuracy, Runtime 1/2 and C++ 1/2 are `not-run` for these profiles.
