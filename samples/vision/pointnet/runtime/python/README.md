English | [简体中文](README_cn.md)

# PointNet Python runtime

<a id="overview"></a>
## Python inference

Segment raw XYZ chair point clouds into four part classes with PointNet.

<a id="directory"></a>
## Directory structure

```text
python/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── cli.py  # Arguments, model selection and result output
├── main.py  # CLI entry: construct model and call predict
├── pointnet.py  # Model initialization and inference stages
└── run.sh  # Run the sample
```

Start with [main.py](main.py): it constructs `PointNetSegmenter` and calls `predict`. [pointnet.py](pointnet.py) contains model initialization and inference stages; [cli.py](cli.py) handles arguments, model selection and result output. Model initialization loads the runtime, so applications can reuse one instance for repeated predictions.

<a id="environment"></a>
## Environment

Python 3.10+, NumPy, PyYAML; matplotlib only for default plots. Actual inference
requires the RDK S100 SDK's `hbm_runtime` in the Python environment supplied by the board image.

```bash
# cwd: repository root; board Python environment
python3 -m pip install numpy PyYAML matplotlib
python3 samples/vision/pointnet/runtime/python/main.py --help
```

<a id="usage"></a>
## Usage

The direct script works from any cwd; model and default input paths are absolute
sample-relative paths. `bash samples/vision/pointnet/runtime/python/run.sh` from
the repository root is an equivalent launcher, forwarding every argument.
```bash
# cwd: repository root; prepare the HBM with model/download.sh first
python3 samples/vision/pointnet/runtime/python/main.py
python3 samples/vision/pointnet/runtime/python/main.py --target s100 --test-pts samples/vision/pointnet/test_data/chair.pts --output-dir outputs/chair --no-plot
python3 samples/vision/pointnet/runtime/python/main.py --dry-run --target s100
```

Exit 0 with labels/JSON written is the CLI success condition; errors print a
message and return 2. Dry-run succeeds without an HBM or SDK but does not validate
actual metadata or inference. Inference never downloads missing artifacts.

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--target` | choice | `auto` | Resolves s100; real execution also verifies local board identity |
| `--asset-id` | string | `None` | Exact S100 reference; required with external model path |
| `--model-path` | string | `None` | Resolves sample `model/s100/pointnet.hbm` |
| `--test-pts` | Path | `samples/vision/pointnet/test_data/chair.pts` | Whitespace XYZ text, one point per row |
| `--output-dir` | Path | `outputs/pointnet` | Relative to cwd; launcher changes cwd to repo root |
| `--no-plot` | flag | `false` | Skip matplotlib and PNGs; retain labels and JSON |
| `--priority` | int | `0` | SDK priority 0–255 |
| `--bpu-cores` | int list | `[0]` | Nonnegative core indexes; hardware availability checked by SDK |
| `--list-models` | flag | `false` | Print manifest records without SDK/file download |
| `--dry-run` | flag | `false` | Resolve preparation only; exclusive with list-models |

`-h` / `--help` prints help and exits. The input default above is resolved to an absolute sample-relative path.

<a id="results"></a>
## Results

`labels.npy`: int32 N-vector, IDs 0=back, 1=seat, 2=leg, 3=arm, same row order as
input. `result.json`: target/asset, input path, point_count, counts, centroid,
radius and observed tensor metadata. `result_orig.png` and `result.png` show
normalized points; the plot helpers in ``cli.py`` preserve the source X/Z/Y axis ordering.
`--no-plot` omits both images. Existing same-named output files are replaced;
choose a separate output directory for each retained experiment.

<a id="integration-example"></a>
## Library integration

The API accepts raw coordinates and normalizes them in `preprocess`. Do not normalize separately or use a file path as the
business input. Loading/plotting belong to the caller.
```python
# cwd: repository root; execute on S100 after model preparation
import numpy as np
from samples.vision.pointnet.runtime.python.cli import resolve_selection, SAMPLE_DIR
from samples.vision.pointnet.runtime.python.pointnet import PointNetSegmenter

points = np.loadtxt(SAMPLE_DIR / "test_data/chair.pts", dtype=np.float32)
selection = resolve_selection("s100")
task = PointNetSegmenter(selection)
prepared = task.preprocess(points)
raw = task.infer(prepared.tensors)
labels = task.postprocess(raw)
print(labels.shape, labels.dtype)
# Equivalent three-stage convenience call:
labels_again = task.predict(points)
# To also keep this call's normalized points and centroid/radius context:
details = task.predict(points, return_details=True)
print(details.prepared.tensors["point"].shape, details.prepared.context.radius)
```

`PointNetPredictionDetails` (opt-in via
`return_details=True`) bundles the labels with this call's prepared record —
the exact normalized `(1,3,N)` tensor plus the frozen centroid/radius context —
so plotting and archiving need no second pass; the default `predict` return
stays the plain labels array and the task never retains a last cloud. The real
runner is lazy, verifies target and
artifact before importing the SDK, and then checks tensor metadata. Use one
runner per inference thread, or serialize calls on a shared runner.

<a id="stage-io"></a>
## Stage IO

| Stage | Input | Output / semantics |
| --- | --- | --- |
| preprocess | finite real ndarray `(N,3)` XYZ | owned contiguous float32 `(1,3,N)`; subtract centroid, divide by maximum Euclidean radius |
| infer | tensor mapping using bound input name | owned raw `(1,N,4)` logits from runtime; no argmax, dequant or IO |
| postprocess | raw tensor with bound shape/dtype | int32 `(N,)` IDs; integer SCALE decoding in float64 before argmax, float32 unchanged |
| predict | raw `(N,3)` points | same stages and labels; `return_details=True` additionally returns this call's prepared record |

The published HBM exposes named `pred` float32 `[1,N,4]` chair-part
logits and auxiliary `trans` float32 `[1,3,3]` XYZ transformation. The
runtime validates both tensors, then decodes `pred` regardless of output
order. A single-logit export is also accepted; other output roles or
transform shapes are rejected.


N comes from compiled metadata and must match exactly; no resampling/padding.
Frozen `prepared.context` stores centroid/radius/count per call and cannot be
overwritten by a later call. Postprocess does not consume it because point order
is unchanged. Integer outputs are affine-decoded in float64 so distinct int8
through int32 raw values keep their ordering for argmax; float32 decoding would
round large integers into artificial ties. Only
exactly equal decoded scores tie, choosing the lowest ID. Integer outputs require
finite positive SCALE metadata; missing/invalid metadata is rejected instead of
guessing.

<a id="troubleshooting"></a>
## Troubleshooting

- Missing model: run the explicit download step in the model guide.
- Target mismatch/no published asset: S100P and S600 cannot reuse the S100 asset.
- Wrong point count/columns: provide exactly N XYZ rows; normals/colors are not inputs.
- Zero-radius/nonfinite cloud: identical or NaN/Inf points cannot be normalized.
- Unknown outputs, differing shapes or unknown dtype: inspect the actual artifact;
  do not bypass binding checks to run a different export.
- Missing matplotlib: install it or use `--no-plot`. The latter still saves labels/JSON.
