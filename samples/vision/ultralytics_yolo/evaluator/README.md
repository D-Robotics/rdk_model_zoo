# Ultralytics YOLO evaluation

English | [简体中文](README_cn.md)

Evaluate a compiled model using the same task implementations as [the Python runtime](../runtime/python/README.md). Detection, instance segmentation and pose use COCO metrics; classification uses ImageNet Top-1/Top-5. YOLO26 OBB exports predictions only. This directory does not download models or datasets, compile models, or measure BPU-only latency.

<a id="dataset"></a>
## Prepare the dataset

Use a matching validation split and preserve the original class order. Dataset acquisition and preparation are documented in [X5 COCO](../../../../platforms/x5/datasets/coco/README.md), [S COCO](../../../../platforms/s/datasets/coco/README.md), [X5 ImageNet](../../../../platforms/x5/datasets/imagenet/README.md) and [S ImageNet](../../../../platforms/s/datasets/imagenet/README.md). Obtain datasets under their own licenses; they are not included in this checkout.

The commands below assume this local layout; replace `/data` and `/models` with your prepared paths:

```text
/data/coco/val2017/000000000139.jpg
/data/coco/annotations/instances_val2017.json
/data/coco/annotations/person_keypoints_val2017.json
/data/imagenet/val/...
/data/imagenet/val.txt
/data/dota/images/...
/models/                         # models compiled for this exact board
```

COCO val2017 contains 5,000 images; instance annotations serve detection/segmentation, while person-keypoints annotations serve pose. With annotations, image names and IDs come from the annotation file. Without annotations, COCO prediction export requires numeric filename stems. `--limit N` selects a subset; `0` means all selected images. Custom class ordering is not supported: standard COCO output-index to category-ID mapping remains in use even when annotations contain only a subset of categories.

ImageNet named labels use `<relative-image-path> <zero-based-class-index>` per line. Alternatively, `--label-file` lists synset IDs in model-class order and filenames must contain the corresponding `n########` identifier. Supply exactly one label source. For the original X5 one-label-per-line validation list, use `--val-format ordered --val-txt FILE --label-offset -1`; the list must match sorted image filenames. Do not apply that offset to already zero-based labels.

<a id="environment"></a>
## Environment

Inference runs on the selected RDK board with its matching `hbm_runtime`, Python, NumPy, OpenCV and SciPy. Use the board-image runtime, not a host compiler package. Keep the full repository checkout. COCO evaluators also import `pycocotools`, including when only exporting predictions:

```bash
python3 -m pip install pycocotools
```

Install this user-space dependency into your intended Python environment; no script installs dependencies automatically. Models must already exist and match the target, task and family. Use [model preparation](../model/README.md) first. `--help` works without loading the board SDK. Dataset runs require hardware; the commands below were checked against the parsers, not executed as new board accuracy measurements.

<a id="command"></a>
## Run an evaluation

All commands use the **repository root** as the working directory. `/models/...` denotes a local file you prepared; the examples do not download it. Use X5 `.bin` or the exact S target's `.hbm`; changing a platform flag cannot convert a model.

### Detection: COCO bounding-box AP

```bash
python3 samples/vision/ultralytics_yolo/evaluator/eval_yolo_det.py \
  --platform x5 --family yolov8 --model-path /models/yolov8n_detect.bin \
  --image-dir /data/coco/val2017 \
  --annotation /data/coco/annotations/instances_val2017.json \
  --conf-thres 0.25 --nms-thres 0.70 --json-save-path /tmp/yolo-det.json
```

### Instance segmentation: COCO box and mask AP

```bash
python3 samples/vision/ultralytics_yolo/evaluator/eval_yolo_seg.py \
  --platform s100 --family yolo11 --model-path /models/yolo11n_seg.hbm \
  --image-dir /data/coco/val2017 \
  --annotation /data/coco/annotations/instances_val2017.json \
  --conf-thres 0.25 --nms-thres 0.70 --json-save-path /tmp/yolo-seg.json
```

### Pose: COCO keypoint AP

```bash
python3 samples/vision/ultralytics_yolo/evaluator/eval_yolo_pose.py \
  --platform s100p --family yolov8 --model-path /models/yolov8n_pose.hbm \
  --image-dir /data/coco/val2017 \
  --annotation /data/coco/annotations/person_keypoints_val2017.json \
  --category-id 1 --conf-thres 0.25 --nms-thres 0.70 \
  --json-save-path /tmp/yolo-pose.json
```

### Classification: ImageNet Top-1/Top-5

```bash
python3 samples/vision/ultralytics_yolo/evaluator/eval_yolo_cls.py \
  --platform s600 --family yolo26 --model-path /models/yolo26n_cls.hbm \
  --image-dir /data/imagenet/val --val-txt /data/imagenet/val.txt \
  --val-format named --label-offset 0 --topk 5 \
  --json-save-path /tmp/yolo-cls.json
```

Use `--topk 5` or larger when reporting Top-5. The current evaluator skips unreadable images and images without usable ground truth; check the output `total` against your expected subset. `total=0` with zero accuracies is not a valid accuracy measurement.

### YOLO26 OBB: prediction export

```bash
python3 samples/vision/ultralytics_yolo/evaluator/eval_yolo_obb.py \
  --platform x5 --family yolo26 --model-path /models/yolo26n_obb.bin \
  --image-dir /data/dota/images --conf-thres 0.25 --nms-thres 0.70 \
  --json-save-path /tmp/yolo-obb.json
```

This writes oriented rectangles and polygon coordinates; it does **not compute DOTA AP**. `--label-path` is accepted for legacy command compatibility but is not scored. Keep the default angle sign/offset unless your reviewed custom output contract requires otherwise.

### Batch evaluation

`eval_batch.py` reads only the immediate files of `--model-dir` and recognizes `_detect_`, `_seg_`, `_pose_`, `_cls_`, `_obb_` filename tokens. For S downloads select the `nash-e`, `nash-m` or `nash-p` directory itself. Use a directory containing only one task/dataset combination: forwarded arguments go to every selected evaluator.

```bash
python3 samples/vision/ultralytics_yolo/evaluator/eval_batch.py \
  --platform x5 --family yolov8 --model-dir /models/coco-detect \
  --image-dir /data/coco/val2017 \
  --annotation /data/coco/annotations/instances_val2017.json --suffix val2017
```

Review the printed selections and confirm interactively; `--yes` skips that prompt. Result JSON is placed beside each model. A full dataset pass may take hours depending on model, board and storage; no duration is guaranteed. Use `--limit 10` for a pipeline check, then remove the limit for full evaluation. A subset result must remain labeled as such.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--platform` | detected board | `x5`, `s100`, `s100p`, `s600`; must match execution hardware |
| `--family` | recognized filename | Explicitly set for custom names; selects the task protocol |
| `--model-path`, `--image-dir` | required | Compiled model and validation images |
| `--annotation` | absent | COCO annotation file; absent means predictions only |
| `--conf-thres` | wrapper default, normally 0.25 | Detection/segmentation/pose/OBB score threshold |
| `--nms-thres` | 0.70 on all targets | Evaluator IoU threshold; S runtime CLI instead defaults to 0.45 |
| `--limit` | 0 | All images, or the first N selected images |
| `--json-save-path` | `results_TASK.json` | Output file relative to caller cwd; create parent directory first |
| `--category-id` | 1 | Pose person category |
| `--val-format`, `--label-offset` | `named`, 0 | Classification label convention |
| `--topk`, `--log-interval` | 5, 1000 | Classification returned classes and progress interval |
| `--angle-sign`, `--angle-offset` | 1, 0 | OBB angle transformation |

Each script's `--help` lists its task-specific contract options. YOLO26 uses direct LTRB; other supported detection families use DFL. S YOLOv10 is NMS-free; X5 YOLOv10 retains NMS. Do not force an unsupported family/task combination or use thresholds to hide an incompatible tensor protocol.

<a id="metrics"></a>
## Metrics and comparability

COCO uses `pycocotools.COCOeval` with `bbox`, `segm` or `keypoints`; the printed AP/AR summary is evaluated only over the selected image IDs. Classification `top1`/`top5` are fractions of processed labeled images, not percentages. OBB JSON is an intermediate result, not an accuracy metric. Elapsed wall time includes Python/data processing and is not BPU inference latency.

Record the source revision, target/system/SDK, model digest, dataset/split, processed count, resize policy and thresholds with any result. A change in any of these can invalidate comparisons with historical benchmark tables. Fixed-image source/unified consistency tests do not establish dataset accuracy.

<a id="outputs"></a>
## Outputs and success criteria

Detection JSON contains COCO `image_id`, `category_id`, `[x,y,width,height]` boxes and scores. Segmentation JSON contains encoded masks; pose JSON contains keypoints and scores. Classification JSON contains `total`, `top1`, `top5`, `elapsed_sec`. OBB entries contain `file_name`, `image_id`, `category_id`, `score`, `rrect` and `polygon`; rectangle angles are radians, polygon coordinates are original-image pixels.

Existing output files are overwritten by the current scripts. Choose a unique output name for each run and preserve stdout containing the COCO summary. Exit 0 means execution completed; empty predictions, no annotations, or classification `total=0` do not constitute a passing accuracy result. COCO empty outputs are written as `[]` and metric computation is skipped explicitly.

<a id="reference-results"></a>
## Reference results and validation scope

The full source benchmark tables remain available locally: [X5 evaluator and benchmarks](../../../../platforms/x5/samples/vision/ultralytics_yolo/evaluator/README.md), [S evaluator and benchmarks](../../../../platforms/s/samples/vision/ultralytics_yolo/evaluator/README.md), [X5 YOLO26](../../../../platforms/x5/samples/vision/ultralytics_yolo26/README.md), [S YOLO26](../../../../platforms/s/samples/vision/ultralytics_yolo26/README.md). Artifact and benchmark facts are maintained in [X5 manifests](../../../../docs/release/x5/) and [S manifests](../../../../docs/release/s/).

Those are historical published records, not measurements of the current unified code. Representative YOLOv8n/YOLO26n detection board comparisons are documented in the [sample guide](../README.md); no new full-dataset accuracy or latency run is claimed here. The current host-only work excludes new board validation.

## Troubleshooting and code navigation

- `pycocotools` missing: install it into the interpreter running the evaluator, including prediction-only COCO runs.
- Ground truth missing or class IDs wrong: check named/ordered/synset format, offset and model class order; do not guess labels from arbitrary filenames.
- COCO filename or ID error: use matching annotation/image splits; numeric stems are required without annotations.
- Empty outputs: inspect one image with the runtime first, check thresholds/task/model, and retain the empty result rather than claiming a metric.
- Output write failure: create a writable parent and use an explicit result path.

`eval_common.py` handles shared arguments, category mapping and dataset selection. Each `eval_yolo_*.py` owns its metric/result format and calls the existing runtime task. `eval_batch.py` only dispatches those commands. Export or tensor-layout changes belong in [conversion](../conversion/README.md) and the runtime binding, not in metric code.

<a id="boundaries"></a>
## Boundaries

These evaluators do not verify model conversion, validate arbitrary custom class orders, or measure end-to-end application performance. Classification skips unreadable/unlabeled images as described above; report the actual processed count. OBB export needs a separate DOTA scorer. Published benchmark rows and prior fixed-image board comparisons are references with their own revisions, not acceptance of every current task/scale. New board and full-dataset runs remain pending.
