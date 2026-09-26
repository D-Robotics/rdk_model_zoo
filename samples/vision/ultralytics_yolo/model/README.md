# Ultralytics YOLO model assets

[简体中文](README_cn.md) · [Sample](../README.md) · [Python runtime](../runtime/python/README.md)

<a id="artifacts"></a>
## Published artifacts

These scripts prepare compiled models for inference; they do not export or compile a training checkpoint. The download URL and exact artifact identity come from the [X5 manifest](../../../../docs/release/x5/models.yaml) or [S manifest](../../../../docs/release/s/models.yaml). Use a complete repository checkout: copying only this directory omits the registry and shared downloader.

| Target | Format | Default destination under this directory | Hardware/compiler family |
|---|---|---|---|
| `x5` | `.bin` | flat files | X5 / Bayes |
| `s100` | `.hbm` | `nash-e/` | S100 / Nash-e |
| `s100p` | `.hbm` | `nash-m/` | S100P / Nash-m |
| `s600` | `.hbm` | `nash-p/` | S600 / Nash-p |

| Family | Published tasks | Scales / restrictions |
|---|---|---|
| `yolo26` | detect, seg, pose, cls, obb | n/s/m/l/x; 25 assets per target |
| `yolov5u` | detect | n/s/m/l/x |
| `yolov8`, `yolo11` | detect, seg, pose, cls | n/s/m/l/x |
| `yolov9` | detect, seg | detection t/s/m/c/e; t unavailable on S600; segmentation c/e, unavailable on S600 |
| `yolov10` | detect | n/s/m/b/l/x |
| `yolo12` | detect | n/s/m/l/x |
| `yolov13` | detect | n/s/l/x; X5 and S100 (S100 uses the iMoonLab source records) |

Publication does not mean that every combination has been board-tested. Do not substitute a different target's model. The canonical X5 full inventory contains 92 assets, including 25 YOLO26 assets; the historical generic X5 wrapper retains its original 67 assets.

<a id="preparation"></a>
## Prepare a model

Run every command below from the **repository root**. The scripts use `python3`, PyYAML (for manifests) and the repository's Python helpers; help and dry-run need no board runtime or network. A real download needs network access and permission to write the destination. Inspect a single-model plan first:

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform x5 --family yolov8 --task detect --model-size n --dry-run
```
The plan prints target, asset count, local path, URL and whether the path exists. To download that selection:

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform x5 --family yolov8 --task detect --model-size n
```
A successful download exits with status 0. To inspect the exact manifest references accepted by the runtime's `--asset-id`:

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform x5 --list-models
```
| Option | Meaning / default |
|---|---|
| `--platform` | x5/s100/s100p/s600; omitted means detect the board. Set explicitly on a host. |
| `--family` | Default yolo11. With `--all`, omission selects all families. |
| `--task` | detect/seg/pose/cls/obb. Omitted: X5 downloads detect/seg/pose/cls; S downloads detect only. |
| `--model-size` | Family default: n except YOLOv9 detection (X5 t, S s). YOLOv9 segmentation defaults to c; c/e may be selected explicitly. |
| `--asset-id` | None; select an exact standalone S source record described below. Cannot combine with `--all`. |
| `--model-dir` | Base directory; defaults to this sample's model directory. |
| `--all` | All published assets, optionally restricted by family; task and size do not filter this mode. |
| `--dry-run` | Plan only; no download or model loading. |

Specify a task when selecting a family that does not publish the platform's whole default task set. Unsupported combinations fail instead of selecting a nearby model. The legacy positional form is still accepted; named options take precedence:

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh s600 yolov8 cls n --dry-run
```
Full inventory downloads can be large. Preview before removing `--dry-run`:

```bash
bash samples/vision/ultralytics_yolo/model/fulldownload.sh \
  --platform x5 --family yolo26 --dry-run
bash samples/vision/ultralytics_yolo/model/fulldownload.sh \
  --platform s100 --dry-run
```
`fulldownload.sh` passes `--all` to the same resolver. It does not need a separate model list.

<a id="accompanying-files"></a>
## Inputs, labels and next steps

The [test data directory](../test_data) contains `bus.jpg` for detection/segmentation/pose, `zebra_cls.jpg` for classification, COCO/ImageNet/DOTA class names, and historical result illustrations. Labels are interpretation aids, not additional model weights; use class order matching your own compiled model. Historical images do not prove a new execution succeeded.

Continue with [Python inference](../runtime/python/README.md), [C++ availability](../runtime/cpp/README.md), [conversion](../conversion/README.md), or [dataset evaluation](../evaluator/README.md). Exported ONNX and training weights are not interchangeable with these board binaries; see conversion for source-model and toolchain requirements.

<a id="local-paths"></a>
## Storage and offline use

The default destination is independent of the shell's working directory. With a custom base directory, S targets still add their march subdirectory:

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform s600 --family yolo26 --task cls --model-size n \
  --model-dir /tmp/rdk-models --dry-run
```
This example resolves below `/tmp/rdk-models/nash-p/`. Remove `--dry-run` to fetch it, then copy the compiled file to the matching board and pass its full path to runtime `--model-path`. Explicit runtime paths are never automatically replaced or downloaded. If a file already exists at the download destination, the downloader verifies it and never overwrites it. Move an unwanted or damaged file aside before retrying.

Historical platform download wrappers retain their original model directories. The canonical runtime uses this sample's model directory; pass `--model-path` to reuse a historical download. Do not rename a `.bin` to `.hbm` or change nash-e/m/p names to simulate another target.

<a id="formats-checksums"></a>
## Formats and integrity

X5 uses a packed NV12 input; S uses separate Y/UV inputs. Non-classification filenames use 640×640. YOLO26 classification filenames use 224×224 on all targets; S600 classification identifiers also use 224×224. S100/S100P v8/v11 classification manifest IDs retain 640×640 compatibility names, while their download URLs use 224×224. A 2026-09-26 read-only HEAD audit found all 20 URL pairs available with equal lengths and ETags; this is not a cryptographic byte comparison or proof of tensor dimensions. Runtime metadata determines actual input geometry; filename tokens are not shape overrides. Existing qualified IDs/local paths are retained.

The downloader rejects empty files and verifies publisher SHA-256 values when recorded. Some manifest entries have no publisher hash: a locally observed digest then identifies bytes but does **not** verify their official origin. Dry-run only checks path existence; `present` is not an integrity check. Downloads use a temporary `.part` file and install the final file only after successful validation. A timeout, HTTP error or unavailable URL is a preparation failure, not permission to fall back to a different artifact.


<a id="standalone-assets"></a>
## Standalone S source identities

The ten existing standalone records are selectable without renaming them into
Ultralytics family records. This is host preparation/routing support; B9 numerical
and C++ consolidation remains under review, and no new board verification is claimed.

| Source sample ID | Task / family | Published targets / sizes | Source Python NMS default |
|---|---|---|---|
| `yolo11` | detect / yolo11 | S100, S600 / n | 0.45 |
| `yolo11_pose` | pose / yolo11 | S100, S600 / n | 0.70 |
| `yolo11_seg` | seg / yolo11 | S100, S600 / n | 0.70 |
| `yolov13_imoonlab` | detect / yolov13 | S100 / n,s,l,x | 0.45 |

Their original URLs stay in the active S manifest. Files are stored under
`model/standalone/<source-sample-id>/<manifest-filename>`; this avoids replacing a
family file with an older source file of the same basename. S600 source filenames
contain `nashe` even under `s600/`; the publication row is preserved, not evidence
that the HBM march has been inspected. No standalone S100P asset is registered.

From the repository root, inspect preparation and inference routing separately:

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform s100 --asset-id s:yolo11_pose:s100/yolo11n_pose_nashe_640x640_nv12.hbm --dry-run
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s100 --task pose \
  --asset-id s:yolo11_pose:s100/yolo11n_pose_nashe_640x640_nv12.hbm --dry-run
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform s100 --family yolov13 --task detect --model-size n --dry-run
```

Remove only the downloader's `--dry-run` to explicitly fetch the chosen model.
The downloader's new `--asset-id` accepts these standalone IDs only and rejects
`--all` or conflicting family/task/size flags. `--all` enumerates the family
inventory (now including S100 iMoonLab), not duplicate YOLO11 source records;
select those with exact IDs. Existing default download sets are unchanged.
The runtime accepts both previously supported family IDs and these source IDs;
pass the correct `--task`. Source IDs retain the table's NMS defaults unless
`--nms-thres` is explicit. Ordinary `--family yolo11` retains the S default 0.45.
Input-image and scheduler defaults remain the unified CLI defaults; pass source
images and `--priority 0 --bpu-cores 0` explicitly for source comparisons.

An explicit `--model-path` keeps that local path and suppresses automatic model
preparation; pairing it with an asset ID declares the expected selection, not a
cryptographic proof of file identity. Do not treat dry-run as SDK validation.
The source detection/segmentation wrappers handle quantized outputs, and the
source pose API returns raw keypoint logits whereas the unified API returns
probabilities. Full numeric/source compatibility is still being audited; see the
[open consolidation record](../../../../docs/releases/unified-migration/2026-09-27-b9-source-consolidation-review.md).
