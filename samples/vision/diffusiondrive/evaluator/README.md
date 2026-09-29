[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive output comparison

This is an offline comparison between float-model references and decoded runtime predictions. It does not run a board, score annotated ground truth, compute NAVSIM PDM Score or approve a model for driving.

<a id="dataset"></a>
## Reference data

The sample retains six deterministic input/reference pairs: the default `reference_inputs.npz` / `reference_outputs.npz`, plus five `case_*` directories. See [test data](../test_data/README.md) for shapes, scene descriptions and historical figures. Each reference contains float32 `trajectory`, `agent_states`, `agent_labels` (logits) and `bev_semantic_map` (logits). These are model outputs, not dataset labels.

Noise is an explicit input. Compare runs made with the same input archive and fixed noise; regenerating noise changes the planning problem. Full NAVSIM evaluation additionally requires scene logs, sensor blobs, maps and metric cache, which this sample does not provide.

<a id="environment"></a>
## Environment

The evaluator needs Python and NumPy; it does not import hbm_runtime or require an HBM. Generate candidate outputs using the [Python runtime](../runtime/python/README.md) on the appropriate target, then copy the complete run directory for offline analysis. The filename `--board-npz` is retained from the source interface; using this option does not itself prove board execution.

<a id="command"></a>
## Commands

From the repository root, compare a saved single-case runtime archive with its corresponding float reference:

```bash
python3 -m samples.vision.diffusiondrive.evaluator.compare_outputs --reference-npz samples/vision/diffusiondrive/test_data/reference_outputs.npz --board-npz outputs/diffusiondrive/outputs.npz --output outputs/diffusiondrive-comparison.json
```

For a batch result, select matching case names on both sides:

```bash
python3 -m samples.vision.diffusiondrive.evaluator.compare_outputs --reference-npz samples/vision/diffusiondrive/test_data/case_017/reference_outputs.npz --candidate-npz outputs/diffusiondrive_cases/case_017/outputs.npz --output outputs/diffusiondrive-case017-comparison.json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--reference-npz` | Required | Float reference archive with the four source output names |
| `--board-npz` / `--candidate-npz` | Required | Decoded six-array runtime `outputs.npz`, not `raw_outputs.npz` |
| `--output` | `null` | Optional new JSON report; otherwise only stdout |

Existing report paths are refused. Return code 0 means valid inputs and completed metric calculation; it does **not** mean the candidate passed an acceptance threshold. Invalid schemas return 2.

<a id="metrics"></a>
## Metrics and validation

Before comparison, both archives must contain exactly the expected names, dtypes and shapes. Float fields must be finite float32; `agent_mask` must be bool and `bev_labels` uint8. Candidate probabilities must be within [0,1]. BEV labels must be in 0..6 and equal the candidate BEV-logit argmax. Shape equality is checked before any NumPy arithmetic, fixing source behavior that could broadcast different-rank label maps into a misleading perfect agreement.

| Reference | Candidate | Comparison |
| --- | --- | --- |
| `trajectory` `[1,8,3]` | `trajectory` | Flattened cosine, MAE, maximum absolute error |
| `agent_states` `[1,30,5]` | `agent_states` | Same tensor metrics |
| `agent_labels` `[1,30]` | `agent_scores` | Apply source clipped sigmoid to reference logits, then tensor metrics |
| `bev_semantic_map` `[1,7,128,256]` | `bev_logits` | Tensor metrics, then compare class argmax maps |

MAE and maximum error use float64 accumulation. Cosine uses flattened float64 vectors; if either norm is zero the JSON value is `null` and explicitly undefined, instead of adding a small denominator and treating the result as an ordinary score. Finite nonzero-vector results are clipped to [-1,1] for roundoff. Reference sigmoid clips logits to [-60,60].

BEV metrics include pixel agreement, class distributions, per-class IoU and macro mean IoU over classes present in either prediction, including background. Absent-in-both classes are excluded. A few rare static-object pixels can materially change macro IoU while barely changing pixel agreement; keep both metrics. Agent masks are schema-checked but not scored: the archive alone does not identify the chosen threshold, so retain the runtime report.

<a id="outputs"></a>
## Report and evidence

JSON includes exact reference/candidate paths and SHA-256, tensor metrics, BEV metrics, zero-norm policy and `status: descriptive; no acceptance threshold`. It explicitly states `dataset_accuracy: false`. Preserve the runtime's `report.json`, physical inputs, raw outputs and image along with the decoded archive. The comparison report binds two files; it does not reconstruct their model/runtime/input provenance if that evidence was discarded.

<a id="reference-results"></a>
## Historical source results

The following records come from the S source documentation and have **not** been rerun during this host migration. Accuracy uses `case_000`, while profiling uses valid quantized `case_017` inputs, one fixed BPU core and 200 frames.

| Metric | S100P | S600 |
| --- | ---: | ---: |
| Trajectory cosine | 0.999857 | 0.999833 |
| Agent-state cosine | 0.996879 | 0.997052 |
| BEV cosine | 0.998913 | 0.998918 |
| BEV pixel agreement | 0.943726 | 0.944061 |
| BEV mean IoU | 0.865501 | 0.868425 |
| Single-thread latency | 14.370 ms | 7.215 ms |
| Single-thread throughput | 69.375 FPS | 138.247 FPS |
| Two-thread average task latency | 28.024 ms | 13.856 ms |
| Two-thread aggregate throughput | 71.109 FPS | 143.767 FPS |
| CPU inference time | 0.0 ms | 0.0 ms |

Source S100P five-case means: trajectory cosine 0.999785, agent-state cosine 0.997986, BEV cosine 0.998799, pixel agreement 0.955664, mean IoU 0.819837. These values are historical comparison results, not thresholds or annotated-dataset accuracy. The source describes BPU-only execution; no new profiling verifies that claim here.

`--thread_num` in historical HRT profiling means concurrent host submission threads, not CPU-core count. Two-thread aggregate throughput must not be converted into single-request latency by reciprocal arithmetic. Source S600 per-case values and the rare-class interpretation are retained in the test-data guide.

<a id="boundaries"></a>
## Verification boundaries

Host tests exercise exact-schema rejection, undefined zero cosine, reference self-comparison and file-hash reporting. Self-comparison only validates the evaluator; it is not quantized-model accuracy evidence. Six-case task/postprocess/render parity is checked against source Python using supplied float arrays. Actual board inference, OE conversion, dataset scoring, full NAVSIM PDM Score and runtime performance remain **not-run**. No default tolerance or release gate is silently inferred from the historical table.
