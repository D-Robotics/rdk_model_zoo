English | [简体中文](README_cn.md)

# RepVGG model artifacts

<a id="artifacts"></a>
## Artifacts

This directory lists the X5 release assets for the single-stage classifier. Choose a target and variant from the sample support matrix.

| Variant | Filename | Target | Format | Source |
| --- | --- | --- | --- | --- |
| `a0` | `RepVGG_A0_224x224_nv12.bin` | x5 | bin | download |
| `a1` | `RepVGG_A1_224x224_nv12.bin` | x5 | bin | download |
| `a2` | `RepVGG_A2_224x224_nv12.bin` | x5 | bin | download |
| `b0` | `RepVGG_B0_224x224_nv12.bin` | x5 | bin | download |
| `b1g2` | `RepVGG_B1g2_224x224_nv12.bin` | x5 | bin | download |
| `b1g4` | `RepVGG_B1g4_224x224_nv12.bin` | x5 | bin | download |

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

cwd: repository root. Shell arguments are positional; the Python form uses named options. Omitting a variant selects `a0` in both forms.

```bash
bash samples/vision/repvgg/model/download.sh x5 a0
python3 samples/vision/repvgg/model/download.py --target x5 --variant b1g4
```

Output: exit 0, filename and observed SHA-256 printed. Download writes a temporary file then installs without overwriting existing bytes. Size/hash failure exits with an error. If the server is unavailable, transfer the matching published file manually into `model/`; preserve its source and observed digest. Do not substitute a different variant.

<a id="accompanying-files"></a>
## Accompanying files

The CLI reads `datasets/imagenet/imagenet_classes.names` for labels and `test_data/gooze.JPEG` as its default image. The API accepts an in-memory image and optional labels; without labels it returns class IDs as strings. No tokenizer or extra model is required.

<a id="local-paths"></a>
## Local paths

Files land at `samples/vision/repvgg/model/<filename>`. The default runtime selection resolves `RepVGG_A0_224x224_nv12.bin` there. An external `--model-path` requires its exact `--asset-id`, e.g. `x5:repvgg:RepVGG_A0_224x224_nv12.bin`.

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
| `timm-a0-rvgg-in1k` | `repvgg_a0_rvgg_in1k_bayese_224x224_nv12.bin` | [`e292d220aa8b`](https://huggingface.co/timm/repvgg_a0.rvgg_in1k/blob/e292d220aa8b811232037f8aa6d6c8c552dbd0c0/README.md) | not-run |
| `timm-a1-rvgg-in1k` | `repvgg_a1_rvgg_in1k_bayese_224x224_nv12.bin` | [`a08d66137071`](https://huggingface.co/timm/repvgg_a1.rvgg_in1k/blob/a08d661370717300608086640790e6a4f54ac802/README.md) | not-run |
| `timm-a2-rvgg-in1k` | `repvgg_a2_rvgg_in1k_bayese_224x224_nv12.bin` | [`87d4d383cb45`](https://huggingface.co/timm/repvgg_a2.rvgg_in1k/blob/87d4d383cb45031cb9fa2fc8ddca73fd6649240f/README.md) | not-run |
| `timm-b0-rvgg-in1k` | `repvgg_b0_rvgg_in1k_bayese_224x224_nv12.bin` | [`cad20ccf7d40`](https://huggingface.co/timm/repvgg_b0.rvgg_in1k/blob/cad20ccf7d400643ebb88cc9403ec97d101e6531/README.md) | not-run |
| `timm-b1g4-rvgg-in1k` | `repvgg_b1g4_rvgg_in1k_bayese_224x224_nv12.bin` | [`54ad9d95acf3`](https://huggingface.co/timm/repvgg_b1g4.rvgg_in1k/blob/54ad9d95acf3a556fa8c9a38a58c4ff0a352666a/README.md) | not-run |

```bash
python3 samples/vision/repvgg/model/download.py --target x5 --variant timm-a0-rvgg-in1k
python3 samples/vision/repvgg/runtime/python/main.py --target x5 --variant timm-a0-rvgg-in1k --dry-run
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
| s100 | `timm-a0-rvgg-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-a0-rvgg-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-a0-rvgg-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-a1-rvgg-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-a1-rvgg-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-a1-rvgg-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-a2-rvgg-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-a2-rvgg-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-a2-rvgg-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-b0-rvgg-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-b0-rvgg-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-b0-rvgg-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-b1g4-rvgg-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-b1g4-rvgg-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-b1g4-rvgg-in1k-s-int16-o1-20261103` | model | 200 |

```bash
python3 samples/vision/repvgg/model/download.py --target s100 --variant timm-a0-rvgg-in1k-s-int16-o1-20261103
python3 samples/vision/repvgg/runtime/python/main.py --target s100 --variant timm-a0-rvgg-in1k-s-int16-o1-20261103 --asset-id s:repvgg:s100/repvgg_a0_rvgg_in1k_nashe_224x224_nv12_float_int16_20261103_o1.hbm --model-path /path/to/repvgg_a0_rvgg_in1k_nashe_224x224_nv12_float_int16_20261103_o1.hbm --test-img samples/vision/repvgg/test_data/gooze.JPEG
```

Substitute the actual file paths. Multi-component tasks require every component together. Historical defaults and measurements retain their original identities. Board smoke, dataset accuracy, Runtime 1/2 and C++ 1/2 are `not-run` for these profiles.
