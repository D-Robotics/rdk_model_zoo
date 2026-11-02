English | [简体中文](README_cn.md)

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

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── download.py  # Prepare model files
├── download.sh  # Model preparation command
└── download_model.sh  # Shell command
```

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

`download_model.sh` accepts the same required target argument. With a null manifest SHA-256, the downloader prints the downloaded file's SHA-256. I/O, selection and download errors return 2.

<a id="accompanying-files"></a>
## Accompanying Files

| File | Role | Required |
| --- | --- | --- |
| `download.py` | Resolves one exact manifest asset and downloads it. | Yes for scripted preparation; no when an HBM is already present. |
| `download.sh` | Explicit target/output-dir shell helper. | No. |
| `download_model.sh` | Launcher delegating to `download.sh`. | No. |
| `../test_data/dog.jpg`, `../test_data/bus.jpg` | CLI inputs for feature and cosine smoke checks. | Only for the demo. |

<a id="local-paths"></a>
## Local Paths

- Default artifact paths: `samples/vision/dinov2/model/nash-e/`, `nash-m/`, or `nash-p/` according to target.
- Runtime selection with no `--model-path` resolves the exact manifest artifact for the concrete target.
- An external `--model-path` must be paired with its exact `--asset-id`, for example `s:dinov2:nash-e/dinov2_vits14_224_int16_nashe.hbm`.

<a id="formats-checksums"></a>
## Formats & Checksums

The three historical publisher SHA-256 values remain unknown; the new converted artifacts have the exact digests listed below. Download digests are computed from the downloaded files.

| Artifact | Format | SHA-256 | Source of value |
| --- | --- | --- | --- |
| `nash-e/dinov2_vits14_224_int16_nashe.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `nash-m/dinov2_vits14_224_int16_nashm.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `nash-p/dinov2_vits14_224_int16_nashp.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |

## Converted artifacts (2026-11-02)

The explicit variant `vits14_224_int16_20261102_o1` currently provides 3 separately compiled O1 HBM artifacts for `s100`, `s100p`, `s600`. The historical default remains selected when `--variant` and `--asset-id` are omitted. These artifacts passed float graph checks, native conversion, and static HBM IO/march inspection. Board smoke, accuracy, and Python/C++ performance checks are **not-run**; historical measurements do not describe these files.

| Target | HBM | SHA-256 |
| --- | --- | --- |
| s100 | `dinov2_vits14_224_int16_nashe_20261102_o1.hbm` | `40d9654edb2b83b12928a4b23e8f482ec30c415d2389af5e64957d9921a9df1a` |
| s100p | `dinov2_vits14_224_int16_nashm_20261102_o1.hbm` | `05a94aae2cb7da14a6a0810c7c7e5e5bff604012b943dd66708e4cd9cc224cbe` |
| s600 | `dinov2_vits14_224_int16_nashp_20261102_o1.hbm` | `e7b720b39d5475d440d8631ba0ac008f230fa28e8a5b1d12d887a60a7cd2c549` |

```bash
python3 samples/vision/dinov2/model/download.py --target s100 --variant vits14_224_int16_20261102_o1
python3 samples/vision/dinov2/runtime/python/main.py --target s100 --variant vits14_224_int16_20261102_o1 --dry-run
```

Only the targets in the table are available for this variant; selecting an unpublished target fails. You can also select the exact `s:dinov2:nash-*/filename.hbm` asset ID. The input remains normalized RGB float32 FEATUREMAP `(1,3,224,224)`. The actual public outputs are float32 and follow the existing metadata-bound `raw_f32` path. Internal quantization retains the int16 recipe; no L2 normalization or softmax is added. Source/weight revisions, full digests, the 50-image calibration recipe, actual environments and validation facts are in the [conversion record](../conversion/converted.json).

## License

The model is quantized from the Apache-2.0 DINOv2 weights published by Meta AI. Preparation code follows the repository [LICENSE](../../../../LICENSE), Apache-2.0.
