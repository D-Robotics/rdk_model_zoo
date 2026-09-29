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

| Artifact | Format | Publisher SHA-256 | Observed SHA-256 (2026-09-24 board runs) |
| --- | --- | --- | --- |
| B0 512 | `.bin` | `null (unknown)` per `docs/release/x5/models.yaml` | `fd184f3559af20eef6a61c3cbd1f99cd13dbd611c3e7271853011ff541d69172` |
| B2 768 | `.bin` | `null (unknown)` per `docs/release/x5/models.yaml` | `adf6436d9c2a4e8374b8124786e19ed9069827489ca7611aeabaee038a33e9ea` |
| B3 896 | `.bin` | `null (unknown)` per `docs/release/x5/models.yaml` | `7044d6c96cdc9cd1afa0d7d73f354c47419d518dcf853caad8123a3d9d304099` |

The observed digests identify the exact bytes downloaded and compared on the X5 8GB/4GB runs ([evidence](../../../../docs/releases/unified-migration/evidence/2026-09-24-b7-other-x5-variants/), [8GB B0 recheck](../../../../docs/releases/unified-migration/evidence/2026-09-24-b7-binding-recheck/)); because the publisher hash is unknown, they do not authenticate origin.
