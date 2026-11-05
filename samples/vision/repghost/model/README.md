English | [简体中文](README_cn.md)

# RepGhost model artifacts

<a id="artifacts"></a>
## Artifacts

All five files are single-stage classifiers in the X5 release manifest. S targets have no RepGhost assets.

| Variant | Filename | Target | Format | Source |
| --- | --- | --- | --- | --- |
| `100` | `RepGhost_100_224x224_nv12.bin` | x5 | bin | download |
| `111` | `RepGhost_111_224x224_nv12.bin` | x5 | bin | download |
| `130` | `RepGhost_130_224x224_nv12.bin` | x5 | bin | download |
| `150` | `RepGhost_150_224x224_nv12.bin` | x5 | bin | download |
| `200` | `RepGhost_200_224x224_nv12.bin` | x5 | bin | download |

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

cwd: repository root. Shell arguments are positional; the Python form uses named options. Omitting a variant selects `100` in both forms.

```bash
bash samples/vision/repghost/model/download.sh x5 100
python3 samples/vision/repghost/model/download.py --target x5 --variant 200
```

Output: exit 0, filename and observed SHA-256 printed. Download writes a temporary file then installs without overwriting existing bytes. Size/hash failure exits with an error. If the server is unavailable, transfer the matching published file manually into `model/`; preserve its source and observed digest. Do not substitute a different variant.

<a id="accompanying-files"></a>
## Accompanying files

The CLI reads `datasets/imagenet/imagenet_classes.names` for labels and `test_data/ibex.JPEG` as its default image. The API accepts an in-memory image and optional labels; without labels it returns class IDs as strings. No tokenizer or extra model is required.

<a id="local-paths"></a>
## Local paths

Files land at `samples/vision/repghost/model/<filename>`. The default runtime selection resolves `RepGhost_100_224x224_nv12.bin` there. An external `--model-path` requires its exact `--asset-id`, e.g. `x5:repghost:RepGhost_100_224x224_nv12.bin`.

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
| `timm-100-in1k` | `repghostnet_100_in1k_bayese_224x224_nv12.bin` | [`3c79eee5f138`](https://huggingface.co/timm/repghostnet_100.in1k/blob/3c79eee5f138318c2cb58957105b91bb50259b29/README.md) | not-run |
| `timm-111-in1k` | `repghostnet_111_in1k_bayese_224x224_nv12.bin` | [`03ffb284f8ea`](https://huggingface.co/timm/repghostnet_111.in1k/blob/03ffb284f8ea7ae1c41a1c2617d4dab98d7cd9f4/README.md) | not-run |
| `timm-130-in1k` | `repghostnet_130_in1k_bayese_224x224_nv12.bin` | [`efb0a7881fad`](https://huggingface.co/timm/repghostnet_130.in1k/blob/efb0a7881fada70cec140ce2810f9c0a7d416704/README.md) | not-run |
| `timm-150-in1k` | `repghostnet_150_in1k_bayese_224x224_nv12.bin` | [`577d935410a6`](https://huggingface.co/timm/repghostnet_150.in1k/blob/577d935410a68c6efc6a0fee9f11f88f942ac690/README.md) | not-run |
| `timm-200-in1k` | `repghostnet_200_in1k_bayese_224x224_nv12.bin` | [`0c520f5e8de7`](https://huggingface.co/timm/repghostnet_200.in1k/blob/0c520f5e8de77f9bfedc23f2104c344b1d685530/README.md) | not-run |

```bash
python3 samples/vision/repghost/model/download.py --target x5 --variant timm-100-in1k
python3 samples/vision/repghost/runtime/python/main.py --target x5 --variant timm-100-in1k --dry-run
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
| s100 | `timm-100-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-100-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-100-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-111-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-111-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-111-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-130-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-130-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-130-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-150-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-150-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-150-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-200-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-200-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-200-in1k-s-int16-o1-20261103` | model | 200 |

```bash
python3 samples/vision/repghost/model/download.py --target s100 --variant timm-100-in1k-s-int16-o1-20261103
python3 samples/vision/repghost/runtime/python/main.py --target s100 --variant timm-100-in1k-s-int16-o1-20261103 --asset-id s:repghost:s100/repghostnet_100_in1k_nashe_224x224_nv12_float_int16_20261103_o1.hbm --model-path /path/to/repghostnet_100_in1k_nashe_224x224_nv12_float_int16_20261103_o1.hbm --test-img samples/vision/repghost/test_data/ibex.JPEG
```

Substitute the actual file paths. Multi-component tasks require every component together. Historical defaults and measurements retain their original identities. Board smoke, dataset accuracy, Runtime 1/2 and C++ 1/2 are `not-run` for these profiles.
