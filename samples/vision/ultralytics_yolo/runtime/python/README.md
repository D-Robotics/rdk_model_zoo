# Python runtime

[简体中文](README_cn.md)

This is the board-side entry point for the shared Ultralytics YOLO sample. It
loads a compiled `.bin` (X5) or `.hbm` (S100/S100P/S600) model through the
`hbm_runtime` supplied by the RDK system image, prepares one BGR image as the
selected target's NV12 input, runs the task decoder, and saves a rendered
result image. The script does not install Python packages. Export and compiler
steps belong in [`conversion/README.md`](../../conversion/README.md).

## Board preparation

Copy or download the artifact for the same target as the board. X5 binds one
packed NV12 input; the S targets bind named NHWC Y and UV inputs. The runtime
reads model metadata and checks batch, geometry, dtype, and output protocol
before inference. A filename suffix is not accepted as a substitute for model
metadata.

The default model path comes from the platform Manifest. On a board, the
historical entry downloads a missing default model; an explicit
`--model-path` is never downloaded. For a host-side path check that does not
load `hbm_runtime`, use `--dry-run` or `--list-models` with an explicit
`--platform`. `--download` prepares the selected Manifest asset and exits.

The board image needs Python 3, NumPy, OpenCV, SciPy, and its matching
`hbm_runtime` module. Model files and the test image must be readable by the
calling user. The runtime never silently falls back to another platform or
model when a requested artifact is missing.

## Shortest detection run

Run from the repository root on the matching board:

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolov8 --task detect \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/yolov8n-x5-detect.jpg
```

For S100/S100P/S600 use the corresponding platform. To use a compiled file
you prepared yourself, pass it explicitly:

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolov8 --task detect \
  --model-path /models/yolo11n_nashp_640x640_nv12.hbm \
  --test-img /data/image.jpg --img-save-path /tmp/yolo-s600.jpg
```

The convenience script accepts the same options after a task name:
`bash runtime/python/run.sh detect --platform s100`. It detects the board when
`--platform` is omitted; explicit target selection is useful for listing,
download, and dry-run on a host. Actual inference rejects an unknown board or
a target mismatch before loading the model.

The result is a tuple-compatible `DetectionResult` containing
`boxes_xyxy` `(N,4)` in original-image pixels, `scores` `(N,)`, and
`class_ids` `(N,)`. The CLI draws boxes and writes the path given by
`--img-save-path` (default `result.jpg`). It prints the selected model, input
protocol, and detections; no result is reported when model loading or binding
fails.

## Task and model selection

`--family` and `--model-size` select a published Manifest combination; omit
the size to use that family/platform's default. `--asset-id` accepts the exact
`group:sample:filename` reference printed by `--list-models`. `--model-path`
selects a local artifact and does not download it; custom filenames should be
paired with `--family` so the decoder is known.

The common task names are `detect`, `seg`, `pose`, `cls`, and `obb` where the
selected family publishes them. YOLOv8/YOLO11 detection uses three feature
levels of DFL logits (`reg=16`). YOLO26 detection uses direct LTRB outputs at
strides 8/16/32 and therefore has a separate binding/decoder. Do not pass a
YOLO26 artifact to a DFL decoder or infer its protocol from output numbering.
The current representative board evidence covers YOLOv8n and YOLO26n
detection; it does not certify every model scale or task.

Useful options include:

```text
--score-thres 0.25       score filter
--nms-thres 0.45         platform default is X5 0.70, S 0.45
--resize-type 0|1        stretch or letterbox; defaults follow the profile
--strides 8,16,32        explicit detection feature strides
--reg 16                 DFL bins for YOLOv8/YOLO11 detection
--priority 0             BPU scheduling priority
--bpu-cores 0            one or more BPU core indexes
```

`--input-shape HxW` is only a fallback for runtimes that do not report input
geometry. A reported geometry that conflicts with it is rejected. `--classes-num`,
`--strides`, `--reg`, `--mc`, and `--nkpt` are model contract parameters, not
ways to make an incompatible artifact load.

## Library entry points

The maintained detection API is importable from `runtime/python`:

```python
from yolo_platform import resolve_platform
from yolo_detect import YoloDetect, YoloDetectConfig

profile = resolve_platform("s600")
config = YoloDetectConfig(
    model_path="/models/yolo11n_nashp_640x640_nv12.hbm",
    platform=profile,
)
detector = YoloDetect(config)
boxes, scores, class_ids = detector.predict(bgr_image)
```

`YoloDetect` accepts an injected runner for host tests and alternate runtime
loaders. The runner is responsible for model execution; geometry preparation,
protocol binding, DFL decode, class-wise NMS, and coordinate restoration remain
in the shared task implementation. `YOLO26Detect` uses the shared image and
runner orchestration with its reviewed direct-LTRB decoder. Legacy X5 and S
modules keep their historical class names and tuple shapes while forwarding to
these maintained paths.

## Code flow

```text
main.py
  -> resolve_target / platform Manifest selection
  -> yolo_dispatch.get_task_types / create_runtime_model
  -> ModelRunner + ModelBinding (input/output contract)
  -> geometry.resize_with_transform + NV12 input binding
  -> YoloDetect or YOLO26Detect decoder + NMS
  -> DetectionResult -> visualize -> --img-save-path
```

`model_binding.py` identifies output roles by reviewed shape/dtype contracts;
compiler enumeration names are opaque. `geometry.py` records the actual
integer resize and padding so inverse boxes use the same transform. The
finite protocols and old-to-new symbol map are in
[`DETECTION_CONTRACT.md`](../../DETECTION_CONTRACT.md).

## Troubleshooting

* **`hbm_runtime` cannot be imported:** use the matching RDK board/system image
  and check its Python module path. Installing a host OpenExplore compiler does
  not provide board runtime support.
* **Unknown board or target mismatch:** omit `--platform` on the board or pass
  the board's real target. A CLI flag cannot override an unknown hardware
  identity.
* **Model not found:** use `--list-models` to see the exact Manifest reference,
  `--download` to prepare a published asset, or provide an existing
  `--model-path`. Explicit paths are never replaced by a similar download.
* **Input/output contract rejected:** verify target, artifact family, static
  even input dimensions, NV12 roles, and the DFL/LTRB output protocol. Check
  `--reg`, `--strides`, and `--classes-num` against the compiled graph.
* **No detections or unexpected boxes:** check the test image color/order,
  `--resize-type`, `--score-thres`, and `--nms-thres`. The decoder restores
  coordinates to the original image; it does not treat output names as a
  geometry declaration.
* **Cannot save the result:** ensure the parent directory of
  `--img-save-path` is writable and that the path is relative to the caller
  when it is not absolute.

Use `python main.py --help` for the complete CLI. `--help`, `--dry-run`,
`--list-models`, and `--download` are host-safe paths that do not run board
inference.
