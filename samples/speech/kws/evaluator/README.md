# KWS evaluation

English | [简体中文](README_cn.md)

<a id="dataset"></a>
## Dataset

The supplied clip is one positive “hey snips” example. Prepare held-out positive and negative clips with stable IDs and the same mono 16 kHz frontend/window policy. Split long recordings explicitly and record the policy with the dataset.

The evaluator consumes saved probabilities. Input JSON schema is `rdk-model-zoo/kws-predictions/v1`, with a `provenance` object containing nonempty `dataset`, `model`, `split` descriptions, and `records` containing unique `id`, integer `label` 0/1 and finite `score` in [0,1]. Include model/input digests and frontend versions in provenance when collecting predictions.

<a id="environment"></a>
## Environment

Python 3.10+ standard library suffices for scoring. No SDK, audio library or NumPy is required. Inference dependencies and metadata gates are listed in the [runtime guide](../runtime/python/README.md).

<a id="command"></a>
## Commands

This example creates two records to show the input schema. Use a new temporary directory:

```bash
KWS_EVAL_DIR=$(mktemp -d)
cat > "$KWS_EVAL_DIR/predictions.json" <<'JSON'
{"schema":"rdk-model-zoo/kws-predictions/v1","provenance":{"dataset":"synthetic documentation example","model":"no model executed","split":"fixture"},"records":[{"id":"positive","label":1,"score":0.9},{"id":"negative","label":0,"score":0.1}]}
JSON
python3 samples/speech/kws/evaluator/evaluate.py \
  --predictions "$KWS_EVAL_DIR/predictions.json" --threshold 0.5 \
  --output "$KWS_EVAL_DIR/metrics.json"
```

For measured predictions replace the input file and provenance with actual captured scores. `--output` is required and must not exist; parent directories are created. `--threshold` defaults to 0.5. Invalid rows, duplicate IDs, empty sets, nonfinite/out-of-range scores and missing provenance fail with exit 2.

<a id="metrics"></a>
## Metrics

Predicted positive means score >= threshold. The report includes TP/TN/FP/FN, accuracy, precision, recall, F1, false accept rate `FP/(FP+TN)` and false reject rate `FN/(FN+TP)`. Undefined denominators produce JSON null. Rates are per supplied clip/window; event rates per hour require event timestamps and aggregation, which the input schema does not carry. The evaluator does not compute an ROC curve.

<a id="outputs"></a>
## Outputs

`metrics.json` uses `rdk-model-zoo/kws-metrics/v1`, preserving provenance and SHA-256 of the exact prediction file. Its schema records `inference_executed=false` and `provenance_independently_verified=false`. The example records produce two correct decisions. Keep the prediction file alongside the report.

<a id="reference-results"></a>
## Source reference results

S100 reference measurements:

| Scope | Source value |
| --- | --- |
| S100 `hrt_model_exec perf`, 100 frames | Average latency 1.176 ms; 830.875 FPS |
| Bundled “hey snips” audio | Approximately 0.985 confidence |

The source performance command was `hrt_model_exec perf --model_file /root/kws/kws.hbm --frame_count 100`. It measures HBM execution for 100 frames; audio loading and feature extraction are outside that command.

<a id="boundaries"></a>
## Boundaries

These metrics describe clip/window decisions. Event rate per hour requires event timestamps and aggregation; board latency is recorded by the S100 runtime workflow. Record the code revision, target, SDK, model/input digests and frontend version with each prediction set.
