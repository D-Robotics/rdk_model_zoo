English | [简体中文](./README_cn.md)

# Model Artifacts — DINOv2 ViT-S/14

<a id="artifacts"></a>
## Artifacts

Each target has its own HBM artifact and Nash march. These are single-model, dual-output graphs exposing `cls_feat` and `patch_feat`.

| Artifact | Format | Target | Stage(s) | Source |
| --- | --- | --- | --- | --- |
| `nash-e/dinov2_vits14_224_int16_nashe.hbm` | HBM | s100 / Nash-E | `cls_feat`, `patch_feat` | download |
| `nash-m/dinov2_vits14_224_int16_nashm.hbm` | HBM | s100p / Nash-M | `cls_feat`, `patch_feat` | download |
| `nash-p/dinov2_vits14_224_int16_nashp.hbm` | HBM | s600 / Nash-P | `cls_feat`, `patch_feat` | download |

Exact manifest URLs:

| Target | Release URL |
| --- | --- |
| s100 | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/dinov2/nash-e/dinov2_vits14_224_int16_nashe.hbm> |
| s100p | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/dinov2/nash-m/dinov2_vits14_224_int16_nashm.hbm> |
| s600 | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/dinov2/nash-p/dinov2_vits14_224_int16_nashp.hbm> |

<a id="preparation"></a>
## Preparation

Run from the repository root and choose the concrete target. No script guesses a target or falls back to S100.

```bash
# cwd: repository root; source: exact manifest URL above
python3 samples/vision/dinov2/model/download.py --target s100
# expect: samples/vision/dinov2/model/nash-e/dinov2_vits14_224_int16_nashe.hbm

# cwd: repository root; alternate target and explicit destination
bash samples/vision/dinov2/model/download.sh s600 /tmp/dinov2-model
# expect: /tmp/dinov2-model/nash-p/dinov2_vits14_224_int16_nashp.hbm
```

`download_model.sh` is a compatibility entry point with the same required target argument. The manifest SHA-256 values are unknown; the downloader prints an observed digest and says it cannot independently verify origin. I/O, selection, or download errors exit 2. This migration did not download an artifact.

<a id="accompanying-files"></a>
## Accompanying Files

| File | Role | Required |
| --- | --- | --- |
| `download.py` | Resolves one exact manifest asset and downloads it. | Yes for scripted preparation; no when an HBM is already present. |
| `download.sh` | Explicit target/output-dir shell wrapper. | No. |
| `download_model.sh` | Source-compatible wrapper delegating to `download.sh`. | No. |
| `../test_data/dog.jpg`, `../test_data/bus.jpg` | CLI inputs for feature and cosine smoke checks. | Only for the demo. |

<a id="local-paths"></a>
## Local Paths

- Default artifact paths: `samples/vision/dinov2/model/nash-e/`, `nash-m/`, or `nash-p/` according to target.
- Runtime selection with no `--model-path` resolves the exact manifest artifact for the concrete target.
- An external `--model-path` must be paired with its exact `--asset-id`, for example `s:dinov2:nash-e/dinov2_vits14_224_int16_nashe.hbm`.

<a id="formats-checksums"></a>
## Formats & Checksums

The active manifest records all three publisher SHA-256 values as unknown. Observed download digests are not origin verification.

| Artifact | Format | SHA-256 | Source of value |
| --- | --- | --- | --- |
| `nash-e/dinov2_vits14_224_int16_nashe.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `nash-m/dinov2_vits14_224_int16_nashm.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `nash-p/dinov2_vits14_224_int16_nashp.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |

## License

The model is quantized from the Apache-2.0 DINOv2 weights published by Meta AI. Preparation code follows the repository [LICENSE](../../../../LICENSE), Apache-2.0.
