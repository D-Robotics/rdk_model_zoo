# Evaluator — FCOS

<a id="dataset"></a>
## Dataset

- Dataset: COCO validation is the historical source reference; the fixed source does not include its version, annotations, or preparation script.
- Smoke input: `samples/vision/fcos/test_data/bus.jpg` is one bundled BGR image, not a COCO evaluation set.

<a id="environment"></a>
## Environment

- Execution target: an identified RDK X5 board with `hbm_runtime` and one exact manifest artifact.
- Host dependencies: Python 3.10+, NumPy, OpenCV, PyYAML, and SciPy from `requirements-host.txt`; SciPy is imported by the fixed source postprocess helper. Host tests use an injected runtime and never load the board SDK.
- The evaluator runs the fixed source wrapper from `platforms/x5/samples/vision/fcos/runtime/python/fcos_det.py` and the unified FCOS task on the same image, artifact, thresholds, and direct-resize geometry.

<a id="command"></a>
## Evaluation Command

The command below creates one new evidence directory and runs the source/unified comparison. It does not download anything. The model must already be present, and the target gate must identify the local board as X5.

```bash
# cwd: repository root; prepare the exact artifact before this step
EVIDENCE="/tmp/fcos-evidence-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$EVIDENCE"
python3 samples/vision/fcos/evaluator/compare.py \
  --target x5 \
  --asset-id x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin \
  --test-img samples/vision/fcos/test_data/bus.jpg \
  --output-dir "$EVIDENCE"
```

The evaluator uses the source default direct resize (`--resize-type 0`). Letterbox inverse geometry is implemented and regression-tested by the runtime task, but the fixed source decoder restores coordinates by direct ratios and therefore is not an evaluator comparison mode.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | str | required | Must be `x5`; local board identity is checked before SDK loading |
| `--asset-id` | str | `None` | Exact manifest identity; `--variant` may select B0/B2/B3 when no asset ID is supplied |
| `--variant` | str | `None` | Omitted selection is B0 |
| `--model-path` | path | `None` | External file, only with the exact asset ID |
| `--test-img` | path | bundled `bus.jpg` | BGR input image |
| `--output-dir` | path | required | Must not already exist; all evidence is written here |
| `--resize-type` | int | `0` | Fixed source direct-resize comparison mode |
| `--conf-thres` | float | `0.5` | Source FCOS confidence threshold |
| `--iou-thres` | float | `0.6` | OpenCV NMS IoU threshold |
| `--priority` | int | `0` | Runtime scheduling priority |
| `--bpu-cores` | int list | `[0]` | Runtime BPU cores |

Return codes are `0` for an exact source/unified match, `1` when both executions complete but any check differs, and `2` for target, input, model, SDK, or evidence-capture failure. A mismatch is never converted to success.

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| input agreement | Packed input names, shape, dtype, and values | Same BGR image and direct resize |
| raw tensor agreement | All 15 tensors by name, shape, dtype, and exact values before dequantization | Same artifact, metadata, image, and runtime target |
| detection agreement | Boxes, scores, and class IDs after source dequant, FCOS decode, and NMS | `conf=0.5`, `IoU=0.6`, exact result arrays |
| COCO mAP | Separate COCO evaluator if supplied externally | Record COCO version, split, annotations, and tolerance |

<a id="outputs"></a>
## Outputs

The new evidence directory contains:

```text
comparison.json       # argv, cwd, UTC times, target, rc, checks, errors, and hashes
errors.json           # exception type/message/traceback when execution fails
input.npy             # exact BGR input passed to both sides
source/metadata.json  source/result.json  source/inputs/*.npy  source/raw/*.npy
unified/metadata.json unified/result.json unified/inputs/*.npy unified/raw/*.npy
```

`comparison.json` records model, input, source/unified code, metadata JSON, and every saved array SHA-256. It also records board identity, exact command arguments, current working directory, UTC start/end, and the evaluator return code. The raw directories contain five classification, five box, and five center-ness arrays for each side.

<a id="reference-results"></a>
## Reference Results

| Metric | Reference | Conditions | Source |
| --- | --- | --- | --- |
| B0/B2/B3 throughput and post-process time | Historical only: B0 323.0 FPS/9 ms, B2 70.9 FPS/16 ms, B3 38.7 FPS/20 ms | Fixed source benchmark conditions; not this host or board run | Fixed source README and evaluator README |
| Host/source numeric agreement | Host synthetic quantized and evaluator-seam tests | No model or board; see current host evidence | [B7 host evidence](../../../../docs/releases/unified-migration/evidence/2026-09-23-b7-fcos-host.json) |
| COCO mAP | not-run | Source supplies no dataset harness or annotations | not-run |

<a id="boundaries"></a>
## Boundaries

- The evaluator does not download models, prepare COCO, measure performance, or claim board success automatically; board evidence remains not-run until the command is actually executed on an identified X5.
- The publisher SHA-256 values for the three manifest rows are unknown, so observed local hashes identify the captured files but do not establish publisher origin.
- Historical screenshots and FPS values are not current measurements.
