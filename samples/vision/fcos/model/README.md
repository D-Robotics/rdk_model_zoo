# Model Artifacts — FCOS

<a id="artifacts"></a>
## Artifacts

| Artifact | Format | Target | Stage | Source |
| --- | --- | --- | --- | --- |
| `fcos_efficientnetb0_detect_512x512_bayese_nv12.bin` | bin | x5 | FCOS detector | manifest download |
| `fcos_efficientnetb2_detect_768x768_bayese_nv12.bin` | bin | x5 | FCOS detector | manifest download |
| `fcos_efficientnetb3_detect_896x896_bayese_nv12.bin` | bin | x5 | FCOS detector | manifest download |

The exact references are `x5:fcos:<filename>`. No model binary is committed in this sample.

<a id="preparation"></a>
## Preparation

```bash
# cwd: repository root
bash samples/vision/fcos/model/download.sh --target x5 --variant efficientnetb0
# expect: one file under samples/vision/fcos/model/; existing files are checked, not replaced

# cwd: repository root
bash samples/vision/fcos/model/fulldownload.sh --target x5
# expect: all three listed files under samples/vision/fcos/model/

# manifest-compatible alias for one selected variant
bash samples/vision/fcos/model/download_model.sh --target x5 --variant efficientnetb2
```

The downloader delegates to the manifest URL and hash fields. The current rows have `sha256: null (unknown)`, so an observed local digest does not establish publisher origin. A missing URL or a non-matching recorded hash fails before installation.

The shell wrappers resolve `download.py` beside themselves and call the board image's `python3`; they do not depend on a repository-local host `.venv`. `download_model.sh` is the compatibility entrypoint registered in `docs/release/x5/models.yaml`.

<a id="accompanying-files"></a>
## Accompanying Files

| File | Role | Required |
| --- | --- | --- |
| `runtime/python/main.py` | Runtime entrypoint and output visualization | yes |
| `test_data/bus.jpg` | Bundled BGR smoke input | no, user input may replace it |
| `datasets/coco/coco_classes.names` | Optional class labels for drawn output | no, numeric IDs are still printed |

<a id="local-paths"></a>
## Local Paths

- Artifacts land in `samples/vision/fcos/model/<manifest filename>`.
- Runtime default model path is the B0 path above; execution still requires `--target x5` and the exact asset ID.
- An external `--model-path` is accepted only together with its exact qualified `--asset-id`; the file name alone is never an identity.

<a id="formats-checksums"></a>
## Formats & Checksums

| Artifact | Format | SHA-256 | Source |
| --- | --- | --- | --- |
| B0 512 | `.bin` | `null (unknown)` | `docs/release/x5/models.yaml` |
| B2 768 | `.bin` | `null (unknown)` | `docs/release/x5/models.yaml` |
| B3 896 | `.bin` | `null (unknown)` | `docs/release/x5/models.yaml` |
