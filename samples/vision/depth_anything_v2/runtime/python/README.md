[English](README.md) | [简体中文](README_cn.md)

# Python runtime

<a id="overview"></a>
## Python inference

Estimate relative depth from a BGR image using Depth Anything V2.

<a id="directory"></a>
## Directory structure

```text
python/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── cli.py  # Arguments, model selection and result output
├── depth_anything_v2.py  # Model initialization and inference stages
├── geometry.py  # Geometry transforms and coordinate restoration
├── main.py  # CLI entry: construct model and call predict
├── run.sh  # Run the sample
└── visualization.py  # Result rendering and image output
```

Start with [main.py](main.py): it constructs `DepthEstimator` and calls `predict`. [depth_anything_v2.py](depth_anything_v2.py) contains model initialization and inference stages; [cli.py](cli.py) handles arguments, model selection and result output. Model initialization loads the runtime, so applications can reuse one instance for repeated predictions.

<a id="environment"></a>
## Environment

Use compatible S100 `hbm_runtime`, Python, NumPy, OpenCV and PyYAML. Listing and
dry-run work on a host without the board SDK. The source delivery used
NumPy 1.26.4, OpenCV 4.11.0.86 and Torch 2.3.1 (installed by its own helper);
this runtime installs nothing automatically. Torch was used by the source only
for resizing and is no longer required.
See [model preparation](../../model/README.md).

<a id="usage"></a>
## Usage

From repository root:

```bash
python -m samples.vision.depth_anything_v2.runtime.python.main --list-models
python -m samples.vision.depth_anything_v2.runtime.python.main --target s100 --dry-run
bash samples/vision/depth_anything_v2/model/download.sh --target s100
bash samples/vision/depth_anything_v2/runtime/python/run.sh --target s100 \
  --test-img samples/vision/depth_anything_v2/test_data/furseal.jpg \
  --output outputs/depth-anything-default
```

Only the last two commands prepare/run a model. Real execution requires S100.
The shell helper changes to repository root; direct Python calls resolve user
paths from current directory. Default image/model paths are sample-relative.
Output directory and any optional image path must not already exist. The optional
color image may not replace any of the five output files.

Optional source letterbox capability is exposed explicitly:

```bash
python -m samples.vision.depth_anything_v2.runtime.python.main --target s100 \
  --resize-type 1 --output outputs/depth-anything-letterbox \
  --img-save-path outputs/depth-anything-letterbox-result.jpg
```

Unlike the source postprocess, this crops letterbox padding before restoring
original dimensions. Use the same mode for reference/candidate comparisons.

<a id="parameters"></a>
## Parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--target` | `auto` | auto selects sole S100 asset; x5/s100p/s600 have no asset and fail |
| `--asset-id` | `null` | exact `s:depth_anything_v2:s100/depth_any.hbm` |
| `--model-path` | `null` | external copy; requires exact asset ID |
| `--test-img` | `samples/vision/depth_anything_v2/test_data/furseal.jpg` | image decoded as BGR uint8 |
| `--output` | `outputs/depth_anything_v2` | new output directory |
| `--img-save-path` | `null` | optional additional source-style color image; new path |
| `--resize-type` | `0` | 0 nearest stretch; 1 linear letterbox with gray127 |
| `--priority` | `0` | integer 0–255; passed to SDK scheduling |
| `--bpu-cores` | `[0]` | nonnegative core IDs; availability depends on SDK |
| `--list-models` | `false` | print matching manifest entries without SDK |
| `--dry-run` | `false` | resolve selection without download or inference |

The default `null` path means derive `model/s100/depth_any.hbm` from the sample.
List and dry-run modes are mutually exclusive. Target selection cannot bypass
actual board identity. Scheduling retains source priority0/core[0]. There is no
implicit warmup or timing. An external path selects a contract, not authenticated
publisher bytes; the manifest has no expected hash.

<a id="results"></a>
## Results

| File | Meaning |
| --- | --- |
| `raw_depth.npy` | unchanged float32 `[1,518,686]` output |
| `depth_native.npy` | restored float32 original H×W relative depth |
| `depth_gray.png` | per-image min/max display normalization to uint8 |
| `depth_color.png` | INFERNO display image |
| `report.json` | model/input digests, target, metadata, runtime version, normalization and scheduling |

The report measures no latency. Unknown runtime version/publisher hash stays
unknown. Negative finite depth values are retained rather than silently treated
as metric distances. A constant map becomes zero grayscale; colorizing zero
still yields the colormap's lowest color, not necessarily black. Invalid/NaN/Inf
values are rejected. Existing outputs are refused to avoid stale-file confusion.

<a id="integration-example"></a>
## Integration example

Run on S100 after model preparation. Image IO remains outside the task:

```python
import cv2
from samples.vision.depth_anything_v2.runtime.python.cli import resolve_selection
from samples.vision.depth_anything_v2.runtime.python.depth_anything_v2 import DepthEstimator

selection = resolve_selection("s100")
task = DepthEstimator(selection, resize_type=0)
image = cv2.imread("samples/vision/depth_anything_v2/test_data/furseal.jpg")
result = task.predict(image)
print(result.depth_native.shape)
# To also keep this call's raw [1,518,686] tensor (the raw_depth.npy contract):
details = task.predict(image, return_details=True)
print(details.raw.shape, details.result.depth_native.shape)
```

`PreparedInput` carries the named physical tensor and immutable `ImageContext`.
`DepthResult` contains owned float `depth_native` and its context.
`DepthPredictionDetails` (opt-in via `return_details=True`) bundles the usual
result with this call's prepared input and raw output so archiving needs no
second inference; the default `predict` return stays the plain `DepthResult`
and the task never retains a last image or last output. Keep the
matching context with every frame; the task stores no last-image dimensions.
Use a separate model instance for each concurrent worker.

The API returns float depth. For a uint8 display, call `visualization.normalize_depth(result.depth_native)` or `colorize_depth(...)`. Keep the frame’s `ImageContext` with its result and synchronize access to any shared runner.

<a id="stage-io"></a>
## Stage contracts

| Stage | Input → output |
| --- | --- |
| `preprocess` | nonempty BGR uint8 HWC → `PreparedInput` with float32 `[1,3,518,686]` |
| `infer` | named input mapping → owned raw float32 `[1,518,686]`, no activation |
| `postprocess` | raw tensor + matching context → float original-size `DepthResult` |
| `predict` | exactly those three stages once; `return_details=True` additionally returns this call's prepared input and raw output; no timing, rendering or IO |


Default input resize is INTER_NEAREST, preserving the actual source helper.
After BGR→RGB, each pixel's three channels use `(rgb - mean(rgb)) /
sqrt(var(rgb) + 1e-5)`, then transpose and cast float32. This is **not ImageNet
mean/std or `/255`**, despite the source docstring. Statistics are computed in float64 from uint8 input before the final float32 cast.

Letterbox uses floor-rounded dimensions, INTER_LINEAR and fill127; gray padding
normalizes to zero. Zero-sized scaled dimensions fail explicitly. Postprocess
crops optional padding and restores with OpenCV INTER_LINEAR. The default stretch
uses the same half-pixel bilinear geometry as the source Torch
`align_corners=False` resize, but floating-point rounding and accumulation can produce small differences.

Metadata must expose exactly one model/input/output with the declared shapes and
float32 types. Internal int16 quantization is not an IO type declaration. Invalid
shape/type, nonfinite values or a mismatched geometry context fail; do not reshape
an incompatible output to make validation pass.

<a id="troubleshooting"></a>
## Troubleshooting

- **S100P or another target rejected:** only S100 has a published asset. The
  source's unconditional S100 shell fallback was removed; do not override identity.
- **Missing SDK/model:** prepare the matching board runtime and model explicitly;
  host dry-run is not an inference test.
- **Wrong metadata:** inspect the actual artifact/runtime version; internal
  quantization prose cannot justify accepting an integer public output.
- **Different colors:** compare float arrays and preprocessing modes first;
  per-image display normalization hides scale/offset differences.
- **Constant map:** zero-gray rendering is defined behavior; a constant
  prediction itself indicates an input/model problem to investigate.
- **Existing output:** choose fresh paths; do not mix partial/stale results.
