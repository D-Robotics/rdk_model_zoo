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

<a id="preparation"></a>
## Preparation

cwd: repository root. Shell arguments are positional; the Python form uses named options. Omitting a variant selects `100` in both forms.

```bash
bash samples/vision/repghost/model/download.sh x5 100
python3 samples/vision/repghost/model/download.py --target x5 --variant 200
```

Success: exit 0, filename and observed SHA-256 printed. Download writes a temporary file then installs without overwriting existing bytes. Size/hash failure exits with an error. If the server is unavailable, transfer the matching published file manually into `model/`; preserve its source and observed digest. Do not substitute a different variant.

<a id="accompanying-files"></a>
## Accompanying files

The CLI reads `datasets/imagenet/imagenet_classes.names` for labels and `test_data/ibex.JPEG` as its default image. The API accepts an in-memory image and optional labels; without labels it returns class IDs as strings. No tokenizer or extra model is required.

<a id="local-paths"></a>
## Local paths

Files land at `samples/vision/repghost/model/<filename>`. The default runtime selection resolves `RepGhost_100_224x224_nv12.bin` there. An external `--model-path` requires its exact `--asset-id`, e.g. `x5:repghost:RepGhost_100_224x224_nv12.bin`.

<a id="formats-checksums"></a>
## Formats and checksums

For every file above: `format: bin`, `sha256: null (unknown)` in `docs/release/x5/models.yaml`. The downloader prints the digest of received bytes; this records local identity and does not independently authenticate the publisher.
