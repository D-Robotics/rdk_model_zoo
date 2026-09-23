# GoogLeNet Python runtime

<a id="environment"></a>
## Environment

Use X5 with its matching `hbm_runtime`, NumPy, OpenCV and PyYAML. Host imports/help/list/dry-run need no board SDK. Host test dependencies: `samples/vision/googlenet/requirements-host.txt`.

[Full prerequisites and tested host versions](../../README.md#prerequisites). Board image and SDK versions remain to be qualified.

<a id="usage"></a>
## Usage

All commands use repository-root cwd. On a host:

```bash
python3 samples/vision/googlenet/runtime/python/main.py --list-models
python3 samples/vision/googlenet/runtime/python/main.py --dry-run --target x5
```

Expected: 1 references or default `googlenet` contract, exit 0. A prepared X5 runs:

```bash
# cwd: repository root
bash samples/vision/googlenet/model/download.sh x5 googlenet
python3 samples/vision/googlenet/runtime/python/main.py \
  --target x5 --variant googlenet \
  --test-img samples/vision/googlenet/test_data/indigo_bunting.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

`run.sh` forwards all runtime options. Download is a separate action.

After preparing the model above, run the default entry on X5 (target detected automatically):

```bash
# cwd: repository root
python3 samples/vision/googlenet/runtime/python/main.py
```

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | choice | auto | execution target: `auto`, `x5`, `s100`, `s100p`, `s600`; an execution target must match detected hardware |
| `--asset-id` | string | null | complete `group:sample:filename` reference from the manifest |
| `--variant` | choice | null | model variant (`googlenet` default when omitted; see `--list-models` for published combinations) |
| `--model-path` | string | null | existing `.bin`; must be paired with `--asset-id`; defaults to the `model/` location for the resolved reference when omitted |
| `--test-img` | string | samples/vision/googlenet/test_data/indigo_bunting.JPEG | BGR input image |
| `--label-file` | string | datasets/imagenet/imagenet_classes.names | one-label-per-line ImageNet labels |
| `--top-k` | int | 5 | number of printed results |
| `--topk` | int | 5 | legacy spelling of `--top-k` |
| `--resize-type` | int | null | `0` direct stretch or `1` letterbox with BGR 127 padding; default follows the bound source (1) |
| `--priority` | int | 0 | runtime scheduling priority (0-255) |
| `--bpu-cores` | int list | [0] | runtime BPU core indexes |
| `--img-save-path` | string | null | optional annotated output image path |
| `--list-models` | flag | false | list manifest-backed references without board access |
| `--dry-run` | flag | false | resolve/check a selection without loading a model or SDK |

<a id="results"></a>
## Results

Prints Top-K class ID/score/label; `ClassificationResult` carries int64 IDs, float32 scores and a label tuple. No file is written without `--img-save-path`. Source-declared score semantics are logits plus softmax; actual board metadata remains to be verified. Equal scores use ascending ID order; this can differ from legacy NumPy tie ordering.

<a id="integration-example"></a>
## Integration example

cwd: repository root on X5, variant googlenet downloaded first. The API never downloads. Labels are optional and omitted here, so their values are class ID strings.

```python
import cv2
from samples.vision.googlenet.runtime.python.classification import ClassificationTask
from samples.vision.googlenet.runtime.python.model_binding import resolve_selection
from samples.vision.googlenet.runtime.python.model_runner import RuntimeModelRunner

selection = resolve_selection("x5", variant="googlenet")
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = ClassificationTask(runner, binding, top_k=5)
image = cv2.imread("samples/vision/googlenet/test_data/indigo_bunting.JPEG")
if image is None:
    raise FileNotFoundError("indigo_bunting.JPEG")
result = task.predict(image)
print(result.class_ids, result.scores, result.labels)
```

<a id="stage-io"></a>
## Stage I/O

| Stage | Input | Output |
| --- | --- | --- |
| pre_process | BGR uint8 H×W×3 | PreparedInput: named flat NV12 uint8 tensor (75264 bytes) and immutable resize context |
| forward | prepared.tensors | runner raw output mapping, unchanged; no softmax or sorting |
| post_process | F32 scores squeezing to (1000,) | softmax and stable Top-K ClassificationResult |
| predict | BGR image | the three stages chained |

Default preprocessing is linear letterbox with BGR 127 padding. Classification postprocessing does not consume geometry. The runner handles SDK loading, scheduling and metadata checks; file/label loading and drawing stay in main.py.

<a id="troubleshooting"></a>
## Troubleshooting

Missing model: prepare it explicitly. `model_path requires --asset-id`: supply the exact qualified reference. Unknown board or target mismatch: use `--dry-run --target x5` on a host, and execute only on matching X5 hardware. S selection: no published asset. Tensor mismatch: retain the actual metadata and check artifact identity; never bypass the binding to force execution.
