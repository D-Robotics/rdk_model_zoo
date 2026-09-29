# YOLO26 Depth Python runtime

<a id="environment"></a>
## Environment

Use the target board's compatible `hbm_runtime` and Python, NumPy, OpenCV and
PyYAML. The SDK is loaded only for inference. Listing assets and resolving a
selection with `--dry-run` work on a host without the SDK. These operations do
not download a model or prove that it executes on a board. See the
[model preparation guide](../../model/README.md) and [sample overview](../../README.md).

<a id="usage"></a>
## Usage

Run the following from the repository root:

```bash
python samples/vision/yolo26_depth/runtime/python/main.py --list-models
python samples/vision/yolo26_depth/runtime/python/main.py --target s600 --variant l --dry-run
python samples/vision/yolo26_depth/model/download.py --target x5 --variant n
python samples/vision/yolo26_depth/runtime/python/main.py --target x5 --variant n --output outputs/depth-x5-n
```

The last command requires an X5 and the prepared model. `run.sh` changes to the
repository root before invoking the same entry point; relative user paths are
therefore relative to that root. Direct Python invocation uses the current
working directory for user paths. Default model and bundled image paths are
resolved from the sample directory. Each output directory must be new.

There are five variants (`n/s/m/l/x`) for each of `x5/s100/s100p/s600`.
Omitting the variant selects `n`, unless an exact asset ID selects another
variant. Conflicting target, variant and asset ID are rejected. Board identity
is checked before production SDK loading; target selection is not permission
to execute a foreign board's model.

<a id="parameters"></a>
## Parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--target` | `auto` | `auto`, `x5`, `s100`, `s100p`, `s600`; default `auto` detects identity |
| `--variant` | `null` | `n/s/m/l/x`; default `n`, or inferred from exact asset ID |
| `--asset-id` | `null` | Exact manifest reference; use `--list-models` to obtain it |
| `--model-path`, `--model` | `null` | External artifact path; requires exact asset ID |
| `--converted-model` | `false` | Explicit user-converted artifact; requires model path and contract asset ID |
| `--test-img`, `--input` | `samples/vision/yolo26_depth/test_data/bus.jpg` | BGR-decodable image; default bundled `test_data/bus.jpg` |
| `--output` | `outputs/yolo26_depth` | New output directory; default `outputs/yolo26_depth` |
| `--warmup` | `3` | Nonnegative number of forward warmups; default 3 |
| `--priority` | `null` | Integer 0–255; S default 0, X5 retains SDK default unless set |
| `--bpu-cores` | `null` | One or more nonnegative core IDs; S default `[0]`, X5 SDK default |
| `--list-models` | `false` | Print manifest assets without loading SDK |
| `--dry-run` | `false` | Resolve selection and boundary without inference or download |

List and dry-run modes are mutually exclusive. An external published X5 model
must match its publisher digest. For a newly compiled model, explicitly use
`--converted-model`: the asset ID selects a tensor contract, not a certificate
for those bytes. Target and metadata checks still apply; the report records
`artifact_origin=user-converted`, `asset_id=null`, the contract reference and
actual file digest. See the [custom-model example](../../model/README.md).
S publisher digests are absent from the source manifest; an observed local hash
does not establish publisher authenticity. Core availability is SDK-dependent.

<a id="results"></a>
## Results

A successful run writes `log_depth.npy` (192×192 float32), `depth_native.npy`
(original image height×width float32), `depth.png`, `overlay.png` and
`report.json`. Lite models additionally write `raw_logit.npy` (192×192 float32).
Depth is relative, not meters. Colors use the source 2nd/98th-percentile,
inverted TURBO visualization and are not an accuracy metric.

The report records model/input hashes, selection, runtime metadata and measured
forward duration. That duration includes runner validation and copying; it is
not isolated BPU latency and excludes preprocessing/postprocessing. Unknown
runtime versions remain `unknown`. Host tests use controlled runtime fixtures;
board accuracy, real SDK execution and dataset metrics remain `not-run` here.
Historical measurements are in the [evaluator guide](../../evaluator/README.md).

<a id="integration-example"></a>
## Integration example

Execute this on the selected board after preparing the model. It does not save
files, warm up or time inference:

```python
import cv2
from samples.vision.yolo26_depth.runtime.python.model_binding import resolve_selection
from samples.vision.yolo26_depth.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.yolo26_depth.runtime.python.yolo26_depth import Yolo26DepthTask

selection = resolve_selection("x5", variant="n")
runner = RuntimeModelRunner(selection)
binding = runner.load()
task = Yolo26DepthTask(runner, binding)
image = cv2.imread("samples/vision/yolo26_depth/test_data/bus.jpg")
prepared = task.pre_process(image)
raw = task.forward(prepared.tensors)
result = task.post_process(raw, prepared.context)
# Equivalently: result = task.predict(image)
print(result.depth_native.shape)
```

`PreparedInput` carries tensors and immutable per-call geometry. `DepthResult`
carries `log_depth`, `depth_native`, optional `raw_logit` and that context.
Keep the matching context with each frame; no mutable last-frame transform is
stored on the task. Runner injection is a host-test seam, not board validation.
The CLI separately sets scheduling parameters; applications may call
`runner.set_scheduling_params(priority=0, bpu_cores=[0])` where supported.
The archived source API and embedded timing are replaced by these three stages
and `predict`; timing, image IO and rendering belong to the caller.

<a id="stage-io"></a>
## Stage contracts

| Stage | Contract |
| --- | --- |
| `pre_process(image)` | Nonempty BGR uint8 HWC → `PreparedInput` |
| `forward(tensors)` | Named physical input → unchanged single float32 SDK output |
| `post_process(raw, context)` | Bound output and matching geometry → owned relative-depth arrays |
| `predict(image)` | Exactly the same three stages, once |

X5 all variants and S `n/s/m` use 768×768 INTER_LINEAR letterbox with fill 114,
then **one flat packed NV12 uint8 array of 884736 bytes**, not separate Y/UV
inputs. Their 192×192 output is already calibrated log depth. Postprocessing
applies exp, resizes to 768, crops padding, and restores original dimensions.
Python ties-to-even rounding determines letterbox geometry; images whose scaled
dimension collapses to zero are explicitly rejected.

S `l/x` use INTER_LINEAR scale-fill, BGR→RGB and `/255`, yielding float32 NCHW
`[1,3,768,768]`. Output is raw logits: clip to `[-4,5]`, apply scale 1 and bias
`-0.2498779296875` (`l`) or `-0.316650390625` (`x`), then exp and restore directly.
Exp and restoration run on CPU. A source prose claim that they are in the graph
is contradicted by its export/runtime code; see the root source audit.

Metadata must describe one model, one input and one float32 output of
`[1,192,192,1]` or `[1,1,192,192]`. Incorrect geometry/type, NaN/Inf, mismatched
context and exp overflow are errors. Do not apply lite calibration twice to a
calibrated log-depth output. Task-level context isolation does not establish
SDK runner thread safety; serialize shared-runner calls unless the SDK warrants
otherwise.

<a id="troubleshooting"></a>
## Troubleshooting

- **Unknown/mismatched board:** run on the selected supported target. Dry-run is
  only selection inspection and cannot establish board compatibility.
- **Missing SDK:** install the matching board runtime using platform guidance;
  NumPy/OpenCV alone cannot run the model.
- **Digest mismatch:** obtain the exact published artifact again. A deliberate
  conversion uses explicit converted mode; do not relabel a corrupt download.
- **Metadata mismatch:** inspect the selected model's input/output contract;
  renaming the file cannot repair layout or raw/log-boundary differences.
- **Output already exists:** choose a new directory to prevent stale files.
- **Invalid image/overflow:** supply a decodable nonempty BGR image and inspect
  model output. Invalid values are rejected rather than silently clipped.
