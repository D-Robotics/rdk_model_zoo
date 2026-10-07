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
| `yolov13` | detect | n/s/l/x; X5 only |

Choose the exact target, family, task and scale from the published inventory; use an artifact compiled for that target.

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

The [test data directory](../test_data) contains `bus.jpg` for detection/segmentation/pose, `zebra_cls.jpg` for classification, COCO/ImageNet/DOTA class names, and result visualizations. Labels help interpret outputs; use the class order matching your compiled model.

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

For a model stored outside the default directory, pass its full path to runtime `--model-path`. Keep the artifact extension and nash-e/m/p directory matched to the target.

<a id="formats-checksums"></a>
## Formats and integrity

X5 uses a packed NV12 input; S uses separate Y/UV inputs. Non-classification filenames use 640×640. YOLO26 classification filenames use 224×224 on all targets; S600 classification identifiers also use 224×224. S100/S100P v8/v11 classification manifest IDs retain 640×640 compatibility names, while their download URLs use 224×224. Runtime metadata determines actual input geometry; filename tokens are not shape overrides.

The downloader rejects empty files and verifies publisher SHA-256 values when recorded. A locally observed digest identifies the downloaded bytes. `--dry-run` reports path presence; it does not download or inspect file contents. Downloads use a temporary `.part` file and install the final file only after successful validation. A timeout, HTTP error or unavailable URL returns a preparation error.

<a id="maintained-scope"></a>
## Runtime output requirements

Runtime `--asset-id` accepts registered Ultralytics YOLO/YOLO26 identities;
prepare assets with the family/task/size commands above. Detection and DFL
segmentation/pose use finite floating outputs with NONE or absent quantization
metadata. Python postprocessing consumes the runtime's floating output; choose an
artifact with the required output type. The runtime checks tensor shape and dtype
at load time. Match the artifact target to the board image.
