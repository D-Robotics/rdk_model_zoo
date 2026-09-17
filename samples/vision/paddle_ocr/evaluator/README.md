English | [简体中文](./README_cn.md)

# PaddleOCR record evaluator

`evaluate.py` compares saved detector/recognizer records. It does not load
`hbm_runtime`, rerun a model, download a dataset, or define a new model
accuracy protocol. This makes it useful after a board run when the input image
set and its labels are available.

## Record format

Each input can be one JSON object, a JSON array, or JSONL with one object per
image. `image` is the preferred stable identifier; `image_id` and `id` are
accepted aliases. If no identifier is present, the record's line/index is
used. Every record must contain aligned `boxes` and `texts` lists. A box is a
polygon represented as `[[x, y], ...]`; the evaluator also accepts one extra
OpenCV contour nesting level (`[[[x, y], ...]]`).

Ground truth example:

```json
{"image":"street-001.jpg","boxes":[[[20,30],[180,30],[180,70],[20,70]]],"texts":["RDK"]}
```

The canonical Python result can be used directly as a prediction object after
adding an `image` field, or as JSONL after one image has been processed per
line:

```json
{"image":"street-001.jpg","target":"s100","boxes":[[[21,31],[179,31],[179,69],[21,69]]],"texts":["RDK"]}
```

## Run

```bash
python samples/vision/paddle_ocr/evaluator/evaluate.py \
  --ground-truth /data/labels.jsonl \
  --predictions /data/predictions.jsonl \
  --iou-threshold 0.5 \
  --output /data/paddleocr-evaluation.json
```

The report is also printed to stdout. For each image, matching walks ground
truth boxes in input order and chooses the highest-IoU unused prediction at or
above the threshold. Equal-IoU candidates keep prediction-file order. Polygon
IoU is computed on each polygon's axis-aligned bounds, matching the box-level
outputs of this sample and keeping the evaluator dependency-free.

The report contains total ground-truth/prediction/match counts, unmatched
counts, detection precision/recall/F1, and a `recognition` object. Recognition
is computed only on IoU-matched regions: `exact_rate` counts exact strings and
`normalized_similarity` is one minus Levenshtein distance divided by the
longer string length (with a minimum denominator of one). If there are no
matched regions, recognition has `status: "not_run"` and both values are zero.
An empty GT record is therefore valid and produces no implicit score.

The evaluator reports measurements for the records supplied by the user. The
bundled demonstration images have no complete labeled corpus, so this tool
does not claim a dataset accuracy number. Runtime/board fidelity evidence is
kept separately in the migration validation report.
