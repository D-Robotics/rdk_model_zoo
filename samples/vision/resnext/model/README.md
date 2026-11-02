English | [简体中文](README_cn.md)

# ResNeXt model artifacts

<a id="artifacts"></a>
## Artifacts

This directory lists the X5 release assets for the single-stage classifier. Choose a target and variant from the sample support matrix.

| Variant | Filename | Target | Format | Source |
| --- | --- | --- | --- | --- |
| `50_32x4d` | `ResNeXt50_32x4d_224x224_nv12.bin` | x5 | bin | download |

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

cwd: repository root. Shell arguments are positional; the Python form uses named options. Omitting a variant selects `50_32x4d` in both forms.

```bash
bash samples/vision/resnext/model/download.sh x5 50_32x4d
python3 samples/vision/resnext/model/download.py --target x5 --variant 50_32x4d
```

Output: exit 0, filename and observed SHA-256 printed. Download writes a temporary file then installs without overwriting existing bytes. Size/hash failure exits with an error. If the server is unavailable, transfer the matching published file manually into `model/`; preserve its source and observed digest. Do not substitute a different variant.

<a id="accompanying-files"></a>
## Accompanying files

The CLI reads `datasets/imagenet/imagenet_classes.names` for labels and `test_data/bee_eater.JPEG` as its default image. The API accepts an in-memory image and optional labels; without labels it returns class IDs as strings. No tokenizer or extra model is required.

<a id="local-paths"></a>
## Local paths

Files land at `samples/vision/resnext/model/<filename>`. The default runtime selection resolves `ResNeXt50_32x4d_224x224_nv12.bin` there. An external `--model-path` requires its exact `--asset-id`, e.g. `x5:resnext:ResNeXt50_32x4d_224x224_nv12.bin`.

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
| `timm-50-32x4d-a1h-in1k` | `resnext50_32x4d_a1h_in1k_bayese_224x224_nv12.bin` | [`cc736cb09401`](https://huggingface.co/timm/resnext50_32x4d.a1h_in1k/blob/cc736cb0940158d05966023116d90fca4c38dae3/README.md) | not-run |

```bash
python3 samples/vision/resnext/model/download.py --target x5 --variant timm-50-32x4d-a1h-in1k
python3 samples/vision/resnext/runtime/python/main.py --target x5 --variant timm-50-32x4d-a1h-in1k --dry-run
```

Exact model URLs/SHA-256/size are in `docs/release/x5/models.yaml`; source and host
receipts are in `conversion/converted.json`. Each weight retains its own model
license in its public release manifest. No board result is implied by download.
<!-- /converted-batch-20261102 -->
