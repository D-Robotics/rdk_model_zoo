# KWS evaluation

English | [简体中文](README_cn.md)

<a id="dataset"></a>
## Dataset

The supplied clip is one positive “hey snips” example, not a labeled evaluation set. Prepare held-out positive and negative clips with stable IDs and the same mono 16 kHz frontend/window policy. A long recording must be split explicitly; record that policy with the dataset. Do not tune and report a threshold on the same split without disclosing it.

The offline evaluator consumes saved probabilities, not audio or HBM files. Input JSON schema is `rdk-model-zoo/kws-predictions/v1`, with a `provenance` object containing nonempty `dataset`, `model`, `split` descriptions, and `records` containing unique `id`, integer `label` 0/1 and finite `score` in [0,1]. Include model/input digests and frontend versions in provenance when collecting real predictions. Provenance is retained, not independently authenticated by the scorer.

<a id="environment"></a>
## Environment

Python 3.10+ standard library suffices for scoring. No SDK, audio library or NumPy is required. Inference dependencies and metadata gates belong to the [runtime](../runtime/python/README.md). This separation allows host scoring of evidence collected elsewhere without silently running a model.

<a id="command"></a>
## Commands

The following host example deliberately writes synthetic scores, demonstrates the schema and reports only arithmetic correctness. It does not measure the published model. Use a new temporary directory:

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

Predicted positive means score >= threshold. The report includes TP/TN/FP/FN, accuracy, precision, recall, F1, false accept rate `FP/(FP+TN)` and false reject rate `FN/(FN+TP)`. Undefined denominators produce JSON null, never an invented perfect score. Rates are per supplied clip/window, **not false accepts per hour**. This evaluator does not compute event timing or an ROC curve.

<a id="outputs"></a>
## Outputs

`metrics.json` uses `rdk-model-zoo/kws-metrics/v1`, preserving provenance and SHA-256 of the exact prediction file. `inference_executed=false` and `provenance_independently_verified=false` make its boundary explicit. The documentation fixture has two correct decisions; it is not a dataset accuracy result. Keep the source prediction file alongside the report.

<a id="reference-results"></a>
## Source reference results

The [archived S evaluator](../../../../platforms/s/samples/speech/kws/evaluator/README.md) records:

| Scope | Source value | Current status |
| --- | --- | --- |
| S100 `hrt_model_exec perf`, 100 frames | Average latency 1.176 ms; 830.875 FPS | Historical only, not rerun |
| Bundled “hey snips” audio | Approximately 0.985 confidence | Historical functional example, not rerun |

The source performance command was `hrt_model_exec perf --model_file /root/kws/kws.hbm --frame_count 100`. It measures runtime performance, not the full audio/file/frontend workflow. Do not derive a new FPS from the reported latency or assert identical measurement scopes. Current host evidence covers feature parity, contracts and metric arithmetic; no new real HBM scores are available.

<a id="boundaries"></a>
## Boundaries

No dataset precision/recall, threshold calibration, board latency, streaming event accuracy or S100P/S600 behavior has been established in this migration. Historical values are not acceptance of the refactored code. A future board comparison must bind exact code, SDK, model/input digests, frontend versions, commands and complete outputs.
