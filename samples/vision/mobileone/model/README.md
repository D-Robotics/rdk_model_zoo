# MobileOne model artifacts

<a id="artifacts"></a>
## Artifacts

Exact X5 release assets; each file is a single-stage classifier. No S asset is published.

| Variant | Filename | Target | Format | Source |
| --- | --- | --- | --- | --- |
| `s0` | `MobileOne_S0_224x224_nv12.bin` | x5 | bin | download |
| `s1` | `MobileOne_S1_224x224_nv12.bin` | x5 | bin | download |
| `s2` | `MobileOne_S2_224x224_nv12.bin` | x5 | bin | download |
| `s3` | `MobileOne_S3_224x224_nv12.bin` | x5 | bin | download |
| `s4` | `MobileOne_S4_224x224_nv12.bin` | x5 | bin | download |

<a id="preparation"></a>
## Preparation

cwd: repository root. Shell arguments are positional; the Python form uses named options. Omitting a variant selects `s0` in both forms.

```bash
bash samples/vision/mobileone/model/download.sh x5 s0
python3 samples/vision/mobileone/model/download.py --target x5 --variant s4
```

Success: exit 0, filename and observed SHA-256 printed. Download writes a temporary file then installs without overwriting existing bytes. Size/hash failure exits with an error. If the server is unavailable, transfer the matching published file manually into `model/`; preserve its source and observed digest. Do not substitute a different variant.

<a id="accompanying-files"></a>
## Accompanying files

The CLI reads `datasets/imagenet/imagenet_classes.names` for labels and `test_data/tiger_beetle.JPEG` as its default image. The API accepts an in-memory image and optional labels; without labels it returns class IDs as strings. No tokenizer or extra model is required.

<a id="local-paths"></a>
## Local paths

Files land at `samples/vision/mobileone/model/<filename>`. The default runtime selection resolves `MobileOne_S0_224x224_nv12.bin` there. An external `--model-path` requires its exact `--asset-id`, e.g. `x5:mobileone:MobileOne_S0_224x224_nv12.bin`.

<a id="formats-checksums"></a>
## Formats and checksums

For every file above: `format: bin`, `sha256: null (unknown)` in `docs/release/x5/models.yaml`. The downloader prints the digest of received bytes; this records local identity and does not independently authenticate the publisher.
