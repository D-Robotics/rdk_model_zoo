English | [简体中文](README_cn.md)

<a id="overview"></a>
## Model preparation

This directory provides the source download for the HobotMot native library and its tracking configurations.

<a id="artifacts"></a>
## Artifacts

| Artifact | Used by | Role |
| --- | --- | --- |
| `vendor/mot-0120205146d88c3474bfc2936522e349e169fa8d/` (extracted from `mot-...tar.gz`) | `runtime/cpp` (both runtimes) | upstream source: wrappers compiled by the sample builds, prebuilt ARM64 `iou_based_mot` core and dependency archives linked from it — identical for every target |
| `configs/iou2.json` | both runtimes | default IOU 2.0 config (no Kalman filter) |
| `configs/iou2_kalman.json` | both runtimes | same defaults with `use_kalman_filter: true` |

<a id="directory"></a>
## Directory structure

```text
model/
├── configs/
│   ├── iou2.json          # default tracker config (min_score 0.5)
│   └── iou2_kalman.json   # Kalman-filter variant
├── download.py            # pinned fetch + SHA-256 verification + extraction
├── download.sh            # download entry point
└── vendor/                # created by download.py, downloaded source
```

<a id="preparation"></a>
## Preparation

From the sample root (`samples/vision/hobot_mot`):

```bash
python3 model/download.py
```

The script downloads
`https://codeload.github.com/D-Robotics/mot/tar.gz/0120205146d88c3474bfc2936522e349e169fa8d`,
verifies `sha256: a0bf3b9738af2f7c4be5885ac777cdf93dab6d4dd374a297ceebebe58268b54b`,
extracts it to `model/vendor/mot-0120205146d88c3474bfc2936522e349e169fa8d/` and
is idempotent (re-running reuses the verified tarball; `--force` re-downloads).
Offline alternative: fetch the same tarball on another machine, check the
SHA-256 and place it at `model/vendor/mot-0120205146d88c3474bfc2936522e349e169fa8d.tar.gz`,
then re-run the script to extract.

<a id="accompanying-files"></a>
## Tracker configuration

| Parameter | Default | Purpose |
| --- | --- | --- |
| `tracker_type` | `"IOU_2.0"` | Tracking algorithm |
| `match_type` | `"IOU"` | Associate boxes by overlap |
| `min_score` | `0.5` | Minimum detection confidence |
| `iou_thres` | `0.2` | IoU association threshold |
| `vanish_frame_count`, `time_gap` | `50`, `40` | Their product is the disappearance timeout in milliseconds |
| `use_kalman_filter` | `false` | Kalman filtering; `true` in `iou2_kalman.json` |

Create a new tracker after changing the configuration. Use the JSON Boolean values `true` or `false` for `use_kalman_filter`.

<a id="local-paths"></a>
## Local paths

The default source directory is `model/vendor/mot-0120205146d88c3474bfc2936522e349e169fa8d/`.
Set `-DHOBOT_MOT_SOURCE_ROOT=<path>` when building to use another directory.
The runtime reads `model/configs/iou2.json` by default. Select another configuration with `--config`.

<a id="formats-checksums"></a>
## Formats and checksums

Source archive: `mot-0120205146d88c3474bfc2936522e349e169fa8d.tar.gz` (gzip tar).
SHA-256: `a0bf3b9738af2f7c4be5885ac777cdf93dab6d4dd374a297ceebebe58268b54b`.

The JSON configuration files in `configs/` are included with the Sample.

