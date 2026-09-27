# YOLOE PF evaluation

English | [简体中文](README_cn.md)

Evaluate a local floating-output model with the same postprocessing used by [the runtime](../runtime/python/README.md). The ONNX backend runs on CPU; the board backend requires the matching installed SDK and a separately prepared float BIN/HBM. This directory does not download models/datasets, compile a model, or reproduce historical BPU performance automatically.

<a id="dataset"></a>
## Dataset and category mapping

Use a held-out COCO-format instance dataset, with `images`, `categories` and `annotations`. Each image needs integer `id`, relative `file_name`, `width` and `height`; each category needs integer `id` and `name`. Box/mask scoring needs valid instance ground truth, including bounding boxes, segmentation, area and crowd flags. Predictions-only mode accepts an image/category manifest with an empty annotations list. Dataset acquisition is described in [X5 COCO](../../../../platforms/x5/datasets/coco/README.md) and [S COCO](../../../../platforms/s/datasets/coco/README.md); datasets are not included here.

PF class IDs are **not COCO category IDs**. Supply a reviewed mapping that binds the fixed [4585-class vocabulary](../test_data/classes.names), checking both source and destination names. [mapping.example.json](mapping.example.json) demonstrates person (PF 2163 → COCO 1) and chair (PF 821 → COCO 62). It covers two categories only and is **not a complete COCO-80 mapping**. Replace/extend it for your annotation categories:

```json
{
  "vocabulary_sha256": "1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3",
  "mapping": [
    {"pf_id": 2163, "pf_name": "person", "category_id": 1, "category_name": "person"},
    {"pf_id": 821, "pf_name": "chair", "category_id": 62, "category_name": "chair"}
  ]
}
```

Every annotation category must be covered; partial category scoring is not silently enabled. PF IDs must be unique, but multiple explicitly named PF classes may map to one dataset category. Such predictions are preserved separately: no additional NMS is applied after mapping. Unmapped PF predictions are excluded from the dataset results and counted in the report. Exact name checks prevent numeric-label drift; they do not prove that a user-chosen semantic mapping is correct.

Images are processed by sorted numeric image ID. `--limit 0` selects all; a positive value selects the first N and labels the metrics as a subset. Duplicate JSON keys/IDs/image paths, paths outside `--image-dir`, unreadable images and annotation/image dimension differences fail. The evaluator does not skip difficult or invalid images. Use a fresh output directory on every run.

<a id="environment"></a>
## Environment

For CPU ONNX evaluation, install the host dependencies into your intended Python 3.10+ environment:

```bash
# cwd: repository root
python3 -m pip install -r samples/vision/yoloe/evaluator/requirements-host.txt
python3 samples/vision/yoloe/evaluator/evaluate.py --help
```

This includes ONNX/ONNX Runtime, NumPy, OpenCV, SciPy, PyYAML and pycocotools; PyTorch is needed only for [checkpoint export](../conversion/README.md). The board route uses the runtime prerequisites and the image-provided `hbm_runtime`; additionally install `pycocotools` there. RLE encoding requires pycocotools even in predictions-only mode. Help works without these packages or the SDK.

The host ONNX backend uses `CPUExecutionProvider` and explicitly disables graph optimization, matching export verification. It receives float RGB NCHW input and **does not simulate an NV12 roundtrip, quantization or BPU execution**. `--target` selects the source preprocessing/postprocessing protocol on the host; it is not a claim that the host is that board. The board backend checks the actual hardware target, file SHA-256 and ten float32 output roles. Original published S quantized models remain incompatible with that float entry.

<a id="command"></a>
## Commands

All commands below run from the repository root. Replace paths and `REPLACE_WITH_64_HEX_SHA256` with your actual local model identity. Use the ONNX digest from `export.json`, or the digest of the explicitly prepared BIN/HBM; the original publication digest cannot identify a different local conversion.

CPU ONNX box/mask evaluation:

```bash
python3 samples/vision/yoloe/evaluator/evaluate.py \
  --backend onnx --target s100 --variant 26n \
  --model-path /work/export26n/yoloe_26n_seg_pf.onnx \
  --model-sha256 REPLACE_WITH_64_HEX_SHA256 \
  --image-dir /data/coco/val2017 \
  --annotation /data/coco/annotations/instances_val2017.json \
  --category-map /data/coco/pf-category-map.json \
  --output-dir /work/evaluation-26n
```

To export predictions without claiming accuracy, add `--predictions-only` and use a new output directory. An annotation-format image/category manifest is still required, but it may have no ground truth. For a smoke check, add `--limit 1`; remove that limit for the intended full split. A subset or predictions-only success is not a full-dataset accuracy result.

Board evaluation, only after preparing a compatible float artifact and entering the matching board environment:

```bash
python3 samples/vision/yoloe/evaluator/evaluate.py \
  --backend board --target s100 --variant 26n \
  --model-path /models/yoloe_26n_float.hbm \
  --model-sha256 REPLACE_WITH_64_HEX_SHA256 \
  --image-dir /data/coco/val2017 \
  --annotation /data/coco/annotations/instances_val2017.json \
  --category-map /data/coco/pf-category-map.json \
  --output-dir /work/evaluation-board-26n
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--backend` | required | `onnx` CPU or `board`; never inferred |
| `--target`, `--variant` | required | X5 11s/m/l; S100 11s or 26n/s/m/l/x; S100P 26n/s/m/l/x |
| `--model-path`, `--model-sha256` | required | Local model and its exact 64-digit SHA-256 |
| `--image-dir`, `--annotation`, `--category-map` | required | Images, COCO JSON and reviewed PF mapping |
| `--output-dir` | required | A new directory; existing content is never overwritten |
| `--limit` | 0 | All images, or first N by sorted numeric image ID |
| `--predictions-only` | false | Write predictions without computing metrics |
| `--score-thres` | 0.25 | Sigmoid confidence cutoff, strictly between 0 and 1 |
| `--nms-thres` | null | E11 uses 0.7 when omitted; E26 rejects an explicit NMS threshold |
| `--resize-type` | 1 | Letterbox; E11 also permits 0 (stretch); E26 requires 1 |
| `--no-morph` | false | Disables S E11's CLI-default mask morphology; other routes never enable it |
| `--max-det` | 300 | E26 Top-K cap, 1..8400; E11 requires the unchanged default value |
| `--multi-label` | false | E26 multiple classes per anchor; E11 rejects it |
| `--threads` | 2 | ONNX CPU threads; nondefault values on the board route are rejected |

Thresholds affect precision/recall. The default matches the demonstration CLI, not an accepted low-threshold COCO accuracy recipe. Record any changes. E11 retains classwise NMS; E26 retains Top-K without NMS. X5 E11 produces full-image masks, S E11/E26 produce ROI masks; all routes reuse the same runtime decoder rather than a separate evaluator algorithm.

<a id="metrics"></a>
## Metrics and comparison

`pycocotools.COCOeval` computes bbox and segmentation AP/AR over the selected image IDs and all annotation categories. AP is a fraction, not a percentage. Reports preserve `-1` for undefined size/category cases rather than replacing it with zero. The actual IoU thresholds, recall threshold count, area labels, image/category IDs and maximum-detection settings are recorded beside each metric. COCO's default maxDets `[1,10,100]` is distinct from the E26 model's default 300-candidate cap.

An empty prediction set is evaluated as empty detections and yields zero AP when ground truth exists; it is not skipped. If selected images have no non-crowd ground truth, metric evaluation fails and asks for predictions-only mode. It cannot manufacture a score from an unlabeled image.

Masks are encoded with original-image geometry. ROI masks must exactly match the clipped, integer-truncated box bounds; full masks must match image height/width. There is no evaluator resizing that could conceal a runtime geometry error. Box and mask predictions are saved separately so COCO segmentation area comes from RLE pixels, not the enclosing box area.

Compare runs only with matching model/checkpoint identities, input images, mapping, target protocol, preprocessing, thresholds and metric settings. Host float output, quantized hardware output and differently rounded Top-K order are distinct comparisons. Per-image wall time covers preprocessing, forward, postprocessing and COCO/RLE encoding, excluding image and JSON file I/O; it is not BPU-only latency or a repeatable performance benchmark.

<a id="outputs"></a>
## Outputs and failures

A successful run exits 0; handled failures exit 2. Initialization errors may occur before output creation. Once dataset execution begins, `evaluation.json` records final status and errors; completed image evidence is flushed incrementally.

| File | Content |
| --- | --- |
| `evaluation.json` | Model/annotation/map identity, configuration, environment, counts, selected IDs, metrics and status |
| `annotations.json`, `category-map.json` | Snapshots of validated inputs, with input and snapshot digests recorded |
| `images.jsonl` | Per-image ID/path/SHA, dimensions, detection/mapping counts and wall time |
| `bbox-predictions.json` | COCO image/category IDs, score and `[x,y,width,height]` |
| `segm-predictions.json` | COCO image/category IDs, score and RLE mask; no bbox field |
| `metrics.log` | Complete COCO scoring output, when metrics are requested |
| `*-predictions.partial.json` | Explicitly incomplete predictions if an image fails mid-run; never scored as a completed run |

Status is `predictions-only`, `evaluated` or `failed`. `metric_scope` distinguishes all annotation images, a selected subset and no requested metric. Inspect processed/selected counts and unmapped predictions before quoting any number. Dataset scoring does not establish an application accuracy acceptance threshold automatically.

<a id="reference-results"></a>
## Historical references and current verification

The following are **original source Runtime-only measurements**, not new measurements of the canonical float route:

| Source model | Board | Runtime latency / FPS | P50 / P95 |
| --- | --- | --- | --- |
| E11s | X5 | 146.16 ms / 6.84 | 144.72 / 152.73 ms |
| E11m | X5 | 177.14 ms / 5.65 | 176.17 / 182.54 ms |
| E11l | X5 | 189.97 ms / 5.26 | 187.99 / 196.30 ms |
| E26n | S100 | 4.943 ms / 200.74 | not published |
| E26s | S100 | 9.944 ms / 100.08 | not published |
| E26m | S100 | 11.765 ms / 84.55 | not published |
| E26l | S100 | 13.417 ms / 74.18 | not published |
| E26x | S100 | 22.013 ms / 45.31 | not published |

X5: RDK X5 V1.0, OS 3.4.1-rp1.0.2, libdnn 1.24.5/HBRT 3.15.55, 1000 MHz, one thread/core_id=1 (BPU core 0), fixed NV12, three rounds of 10 warmup plus 200 timed frames. The two-thread 11s test failed with an ION allocation error. See the [full X5 source record](../../../../platforms/x5/samples/vision/yoloe/evaluator/README.md).

S100: V1P0, OS 4.0.5-Beta, UCP 3.13.6/HBRT 4.7.5, OE 3.7.0 INT8 KL, 2026-09-08, 200 frames/warmup, thread_num=1/core_id=0. S100P performance was not measured. The [S26 source record](../../../../platforms/s/samples/vision/yoloe26_seg/evaluator/README.md) separately records original n-model S100P Python/C++ pixel-exact mask comparison on one image; that is not current code, other sizes or dataset mAP. The [S11 source evaluator](../../../../platforms/s/samples/vision/yoloe11_seg/evaluator/README.md) was a placeholder, with no accepted metrics to migrate.

Current real ONNX predictions are host checks on the bundled image with an explicitly generated PF-category manifest and **no ground truth**. They are not COCO AP measurements. The E26 PT files used for recent export have different file hashes from the archived release sidecars; that does not prove changed tensors by itself, but prevents claiming the same original checkpoint baseline. In particular, float detection counts must not be presented as reproduced INT8 counts or attributed solely to quantization. Exact evidence is in the [evaluation implementation record](../../../../docs/releases/unified-migration/2026-09-28-yoloe-evaluation-review.md).

<a id="boundaries"></a>
## Boundaries and checks

No actual board/SDK/OE evaluation, held-out dataset mAP or new performance benchmark has run in this migration increment. The board backend is implemented but unverified on hardware. CPU RGB evaluation excludes NV12 conversion effects. The fixed vocabulary, mapping and model bytes must stay paired; arbitrary vocabulary edits do not teach new categories.

```bash
# cwd: repository root; evaluator host dependencies installed
python3 -m unittest discover -s samples/vision/yoloe/evaluator/tests
python3 -m unittest discover -s samples/vision/yoloe/tests
```

Tests cover real COCO scoring of known synthetic masks, empty predictions, explicit mapping, strict image/mask geometry, missing-file partial failure and predictions-only boundaries. Synthetic perfect AP verifies the scorer only; it is not a YOLOE result. Historical source runtime-contract tests remain archived; canonical runtime tests cover the shared stages. Native C++ migration and whole-branch independent acceptance remain separate work.
