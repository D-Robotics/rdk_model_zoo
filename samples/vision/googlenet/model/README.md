English | [简体中文](README_cn.md)

# GoogLeNet model artifacts

<a id="artifacts"></a>
## Artifacts

The historical X5 assets are listed first. New S HBM profiles, exact identities and native commands are listed below. Choose the actual target and its explicit variant.

| Variant | Filename | Target | Format | Source |
| --- | --- | --- | --- | --- |
| `googlenet` | `googlenet_224x224_nv12.bin` | x5 | bin | download |

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

cwd: repository root. Shell arguments are positional; the Python form uses named options. Omitting a variant selects `googlenet` in both forms.

```bash
bash samples/vision/googlenet/model/download.sh x5 googlenet
python3 samples/vision/googlenet/model/download.py --target x5 --variant googlenet
```

Output: exit 0, filename and observed SHA-256 printed. Download writes a temporary file then installs without overwriting existing bytes. Size/hash failure exits with an error. If the server is unavailable, transfer the matching published file manually into `model/`; preserve its source and observed digest. Do not substitute a different variant.

<a id="accompanying-files"></a>
## Accompanying files

The CLI reads `datasets/imagenet/imagenet_classes.names` for labels and `test_data/indigo_bunting.JPEG` as its default image. The API accepts an in-memory image and optional labels; without labels it returns class IDs as strings. No tokenizer or extra model is required.

<a id="local-paths"></a>
## Local paths

Files land at `samples/vision/googlenet/model/<filename>`. The default runtime selection resolves `googlenet_224x224_nv12.bin` there. An external `--model-path` requires its exact `--asset-id`, e.g. `x5:googlenet:googlenet_224x224_nv12.bin`.

<a id="formats-checksums"></a>
## Formats and checksums

For every file above: `format: bin`, `sha256: null (unknown)` in `docs/release/x5/models.yaml`. The downloader prints the computed digest after transfer; the manifest SHA-256 is `null (unknown)`.

<!-- converted-all-20261103 -->

## Converted S profiles (2026-11-03)

These independent profiles passed conversion, complete OE collection and actual HBM metadata binding. Source weights, graphs, per-target recipes, calibration counts and digests are in [converted_targets_20261103.json](../conversion/converted_targets_20261103.json).

| Target | Native variant | Components | Calibration photographs |
| --- | --- | --- | ---: |
| s100 | `torchvision-021-crop224-int16-o1-20261103` | model | 200 |
| s100p | `torchvision-021-crop224-int16-o1-20261103` | model | 200 |
| s600 | `torchvision-021-crop224-int16-o1-20261103` | model | 200 |

```bash
python3 samples/vision/googlenet/model/download.py --target s100 --variant torchvision-021-crop224-int16-o1-20261103
python3 samples/vision/googlenet/runtime/python/main.py --target s100 --variant torchvision-021-crop224-int16-o1-20261103 --asset-id s:googlenet:s100/googlenet_tv021_crop224_nashe_int16_20261103_o1.hbm --model-path /path/to/googlenet_tv021_crop224_nashe_int16_20261103_o1.hbm --test-img samples/vision/googlenet/test_data/indigo_bunting.JPEG
```

Substitute the actual file paths. Multi-component tasks require every component together. Historical defaults and measurements retain their original identities. Board smoke, dataset accuracy, Runtime 1/2 and C++ 1/2 are `not-run` for these profiles.
