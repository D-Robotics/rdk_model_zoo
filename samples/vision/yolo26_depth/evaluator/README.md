[English](README.md) | [简体中文](README_cn.md)

# Offline depth evaluation

These tools prepare evaluation inputs and compare **saved arrays**. They do not
load a BIN/HBM or run a board model. Runtime, dataset accuracy and latency are
separate claims. This migration has host fixture tests, not new board or SUNRGBD
measurements.

<a id="dataset"></a>
## Dataset and input identity

Supply locally obtained SUN RGB-D images and metric-depth `.npy` arrays. The
repository does not include the dataset. Prepare a source JSON manifest with
`records`, for example:

```json
{
  "records": [
    {"index": 7, "sensor": "sensor_name", "image": "images/0007.png",
     "image_hw": [480, 640], "depth_m": "depth/0007.npy"}
  ]
}
```

Image and depth paths are relative to `--source-root`. `image_hw` is checked
when present. Existing nonnegative integer record IDs are preserved; if omitted,
preparation assigns source-list positions. Duplicate IDs fail. Missing depth can
be retained during input preparation but cannot be evaluated against ground truth.

Three source protocols are retained:

| Protocol | Prepared input | Restoration / ground truth |
|---|---|---|
| `deployment_letterbox` | RGB CHW uint8 `.bin`, 768 letterbox with 114 padding | calibrated log-depth → exp → resize 768 → crop → original H×W; original GT |
| `deployment_scale_fill` | RGB NCHW float32 `/255` `.npy`, direct 768 stretch | raw logit → clip/scale/bias → exp → original H×W; original GT |
| `ultralytics_validator` | RGB CHW uint8 `.bin`, source long-side resize then square stretch | calibrated log-depth → exp → 768 square; GT uses the same two sizes with nearest interpolation |

The RGB binaries are prepared model inputs, **not packed runtime NV12**.
Normalization and model input adaptation must match the producer of saved
outputs. Do not compare arrays generated with different protocols merely because
their shapes agree.

<a id="environment"></a>
## Environment

The default host path needs Python, NumPy and OpenCV. It imports no board SDK.
Dataset metrics accumulate in float64 with source formulas. `--resize-backend
torch` optionally retains source evaluator interpolation (`align_corners=False`)
and loads Torch only when used. The default OpenCV interpolation matches the
canonical runtime. Numerical summation is not claimed bit-identical to source
Torch float32 reductions. The optional Torch backend has not been executed in
this migration environment.

Run commands below from this `evaluator/` directory. Write datasets, reports and
images outside the sample tree. Each preparation/comparison output directory and
evaluation report must be new.

<a id="command"></a>
## Commands and saved-output format

```bash
python prepare_sunrgbd.py --source-root /work/sunrgbd \
  --source-manifest /work/sunrgbd/source.json --output /work/depth/prepared \
  --size 768 --screen-count 20 --screen-seed 20260726
```

By default all three protocols are generated; repeat `--protocol` to select a
subset and reduce storage. Every tensor has a recorded hash, geometry and dtype.
The screen subset is deterministic and stratified by sensor; requesting fewer
samples than sensors or zero samples is supported. Screen selection is metadata,
not an instruction for the evaluator to silently discard other output records.

Produce floating and compiled-model outputs separately in their matching
export/runtime environments. Their inference is not performed by these tools.
Save two NPZ files with `indices` (one-dimensional unique integer array) and one
value array indexed in that order:

| Boundary | Reference key | Candidate key | Per-record shape |
|---|---|---|---|
| `log` | `float_log` | `quant_log` | `[192,192]`, `[1,192,192,1]` or `[1,1,192,192]` |
| `raw` | `float_raw` | `quant_raw` | same shapes |

Values must be finite floating-point outputs before depth restoration. Runtime
`log_depth.npy` is suitable for a **log** record; lite `raw_logit.npy` is suitable
for a **raw** record. `depth_native.npy` is already decoded and cannot be placed
in either output slot. The following example packages one existing pair; it does
not generate reference inference:

```python
import numpy as np
reference = np.load('/work/depth/reference_log_depth.npy', allow_pickle=False)
candidate = np.load('/work/depth/candidate_log_depth.npy', allow_pickle=False)
np.savez('/work/depth/float.npz', indices=np.array([7], dtype=np.int64),
         float_log=reference[None])
np.savez('/work/depth/quant.npz', indices=np.array([7], dtype=np.int64),
         quant_log=candidate[None])
```

Use the actual prepared record ID, not an invented position. Reference and
candidate ID sets must match exactly and exist in the prepared manifest.

```bash
python eval_sunrgbd.py --prepared-manifest /work/depth/prepared/manifest.json \
  --source-root /work/sunrgbd --float-outputs /work/depth/float.npz \
  --quant-outputs /work/depth/quant.npz --candidate-name x5-n-candidate \
  --protocol deployment_letterbox --boundary log --variant n \
  --report /work/depth/evaluation.json
```

For S n/s/m the same letterbox/log combination applies. For S l/x use
`deployment_scale_fill`, `--boundary raw`, the matching `--variant` and raw NPZ
keys. Experimental S n/s/m lite outputs can use that raw protocol with the source
coefficients, but are not published default runtime profiles. `--boundary log`
never applies those coefficients again. The variant/geometry declarations must
match the actual producer; the evaluator cannot independently infer that history
from arbitrary saved arrays.

To compare two **restored original-size** depth maps and render common-range
panels:

```bash
python eval_numeric.py --image /work/depth/input.png \
  --official /work/depth/reference_depth.npy --candidate /work/depth/candidate_depth.npy \
  --reference-name fp32-reference --candidate-name s600-l-candidate \
  --output /work/depth/single-image
```

`--x5` remains an alias for `--candidate`; the candidate is no longer hardcoded as
X5 n. `--official` names the supplied reference path, not an authenticity check.
Both maps must match the image shape and contain finite values greater than 1e-6.

<a id="metrics"></a>
## Metrics and alignment

Dataset evaluation reports delta1/2/3 thresholds `1.25`, `1.25²`, `1.25³`, AbsRel,
RMSE and SILog. GT is valid only for finite `0.001 < depth < 100`; predictions are
clamped to `[0.001,100]` after optional per-image alignment. Metrics pool pixels
across selected images, not unweighted per-image averages.

Both unaligned and median-aligned metrics are retained. Dataset alignment uses
the **lower median** for an even-length sample, matching source `torch.median`.
The single-image comparison preserves source NumPy's **average of the two middle
values** instead. These policies can produce different scales; the reports name
them explicitly.

Raw-domain and restored-depth fidelity report MAE, RMSE, max absolute error,
relative error and cosine. Empty/nonfinite/mismatched arrays fail rather than
silently dropping values. Zero-norm cosine is `null`. No-valid-GT accuracy is
`null` with `valid_pixels=0`, not a perfect score. No tool automatically declares
a candidate accepted from cosine alone.

<a id="outputs"></a>
## Output artifacts

Preparation produces per-protocol tensors and `manifest.json`, including source
manifest/image/tensor hashes, record IDs, geometry and screen selection.
Dataset evaluation writes one JSON report with both models' GT metrics,
candidate-minus-reference deltas, raw/depth fidelity, per-image results and input
archive hashes. Its board field states that this offline tool did not run a board.

Single-image comparison writes `comparison-report.json`,
`candidate_depth_median_aligned.npy`, common-range depth/overlay PNGs, absolute
relative error PNG and `comparison.jpg` (1620×586 contact sheet). Visualization
uses the reference 2nd/98th percentile range for both maps; colors are not metres.
Names now use reference/candidate instead of incorrectly labeling every input X5.

<a id="reference-results"></a>
## Historical source results — not remeasured

X5 source records OE 1.2.8 / Mapper 1.24.3, 768 input, max percentile 0.9999,
O3 latency and int16 tail convolution. HRT numbers cover model execution only:

| Variant | X5 1-thread latency ms | 1-thread FPS | 2-thread total FPS |
|---|---:|---:|---:|
| n | 23.194 | 43.085 | 45.682 |
| s | 36.168 | 27.637 | 28.615 |
| m | 60.783 | 16.449 | 16.751 |
| l | 75.336 | 13.272 | 13.470 |
| x | 161.022 | 6.210 | 6.253 |

S evaluator source reports the following HRT latency table in milliseconds:

| Variant | S100 | S100P | S600 |
|---|---:|---:|---:|
| n | 3.165 | 2.254 | 1.760 |
| s | 4.490 | 3.244 | 2.363 |
| m | 8.246 | 5.986 | 4.062 |
| l | 9.790 | 7.090 | 4.881 |
| x | 19.059 | 12.853 | 9.097 |

Its example selects lite n, while the release uses NV12 n/s/m. The source does
not bind every table row to an artifact hash, so this table is not relabeled as
confirmed published-profile timing. Its claimed log cosine 0.9985–0.9998 also
differs from the root's mixed-profile table; the root's s=0.9984 contradicts its
all-pass threshold of 0.999. These conflicts are retained in the
[source audit](../../../../docs/releases/unified-migration/2026-09-26-b8-yolo26-depth-source-review.md).
No SUNRGBD accuracy result or new acceptance is implied by these tables.

<a id="boundaries"></a>
## Verification boundaries

Host fixtures cover correct raw/log decoding, canonical runtime parity, lower
median and pixel pooling, invalid data/ID rejection, three preparation protocols,
small screen selection and real report/image writes. They do not establish
Torch-backend parity, model output accuracy, OE compilation, board behavior or
performance. Optional Torch interpolation, dataset runs and all new board results
remain not-run. Raw output arrays and their hashes do not by themselves prove
which model, input preprocessing or board produced them; retain producer evidence
alongside these reports.
