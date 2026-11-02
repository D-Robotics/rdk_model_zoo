English | [简体中文](README_cn.md)

# RepViT model artifacts

<a id="artifacts"></a>
## Artifacts

This directory lists the X5 release assets for the single-stage classifier. Choose a target and variant from the sample support matrix.

| Variant | Filename | Target | Format | Source |
| --- | --- | --- | --- | --- |
| `m0_9` | `RepViT_m0_9_224x224_nv12.bin` | x5 | bin | download |
| `m1_0` | `RepViT_m1_0_224x224_nv12.bin` | x5 | bin | download |
| `m1_1` | `RepViT_m1_1_224x224_nv12.bin` | x5 | bin | download |

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

cwd: repository root. Shell arguments are positional; the Python form uses named options. Omitting a variant selects `m0_9` in both forms.

```bash
bash samples/vision/repvit/model/download.sh x5 m0_9
python3 samples/vision/repvit/model/download.py --target x5 --variant m1_1
```

Output: exit 0, filename and observed SHA-256 printed. Download writes a temporary file then installs without overwriting existing bytes. Size/hash failure exits with an error. If the server is unavailable, transfer the matching published file manually into `model/`; preserve its source and observed digest. Do not substitute a different variant.

<a id="accompanying-files"></a>
## Accompanying files

The CLI reads `datasets/imagenet/imagenet_classes.names` for labels and `test_data/yurt.JPEG` as its default image. The API accepts an in-memory image and optional labels; without labels it returns class IDs as strings. No tokenizer or extra model is required.

<a id="local-paths"></a>
## Local paths

Files land at `samples/vision/repvit/model/<filename>`. The default runtime selection resolves `RepViT_m0_9_224x224_nv12.bin` there. An external `--model-path` requires its exact `--asset-id`, e.g. `x5:repvit:RepViT_m0_9_224x224_nv12.bin`.

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
| `timm-m0-9-dist-450e-in1k` | `repvit_m0_9_dist_450e_in1k_bayese_224x224_nv12.bin` | [`c073717dd84b`](https://huggingface.co/timm/repvit_m0_9.dist_450e_in1k/blob/c073717dd84b5af671dd51b187b99ac55b2af4f0/README.md) | not-run |
| `timm-m1-0-dist-450e-in1k` | `repvit_m1_0_dist_450e_in1k_bayese_224x224_nv12.bin` | [`0e20f9f1bfa9`](https://huggingface.co/timm/repvit_m1_0.dist_450e_in1k/blob/0e20f9f1bfa9772e63b813835bab2d6f0d168f89/README.md) | not-run |
| `timm-m1-1-dist-450e-in1k` | `repvit_m1_1_dist_450e_in1k_bayese_224x224_nv12.bin` | [`ff441038a429`](https://huggingface.co/timm/repvit_m1_1.dist_450e_in1k/blob/ff441038a42982fff1c8c75be7c794f9ba76db72/README.md) | not-run |

```bash
python3 samples/vision/repvit/model/download.py --target x5 --variant timm-m0-9-dist-450e-in1k
python3 samples/vision/repvit/runtime/python/main.py --target x5 --variant timm-m0-9-dist-450e-in1k --dry-run
```

Exact model URLs/SHA-256/size are in `docs/release/x5/models.yaml`; source and host
receipts are in `conversion/converted.json`. Each weight retains its own model
license in its public release manifest. No board result is implied by download.
<!-- /converted-batch-20261102 -->
