# ViT evaluation

<a id="dataset"></a>
## Dataset

Ten bundled CIFAR-10 images (one per class) support functional checks, not full-dataset accuracy. A full CIFAR-10 evaluation set is not included, and no dataset accuracy evaluator is shipped by this source. The historical subset/version/protocol is not fully recorded; do not treat the ten images as that benchmark.

<a id="environment"></a>
## Environment

[Runtime prerequisites](../runtime/python/README.md#environment). Host tests use an injected runner and real preserved preprocessing helpers. Board comparison needs S100 and both original/unified runtime dependencies; no OE required.

<a id="command"></a>
## Commands

```bash
# cwd: repository root; expected: unittest OK, exit 0
python3 -m unittest discover -s samples/vision/vit/tests -v
```

Board recipe below has not run. Prepare the model, then compare on the same board/input/artifact. Repeat with int16 and all ten bundled images; duration unmeasured.

```bash
# cwd: repository root
bash samples/vision/vit/model/download.sh s100 int8
```

```bash
# cwd: repository root on S100; download int8 first
PYTHONPATH="$PWD:$PWD/platforms/s:$PWD/platforms/s/samples/vision/vit/runtime/python" python3 - <<'PYTHON'
from pathlib import Path
from datetime import datetime, timezone
import cv2
import numpy as np
from vit import ViT, ViTConfig
from samples.vision.vit.runtime.python.model_binding import resolve_selection
from samples.vision.vit.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.vit.runtime.python.classification import ClassificationTask
variant = "int8"  # repeat with int16 after preparing its artifact
selection = resolve_selection("s100", variant=variant)
image = cv2.imread("samples/vision/vit/test_data/airplane_0000.png")
if image is None:
    raise FileNotFoundError("airplane_0000.png")
legacy = ViT(ViTConfig(str(selection.model_path), resize_type=0))
legacy.set_scheduling_params(priority=0, bpu_cores=[0])
old_raw = legacy.forward(legacy.pre_process(image))
old_top = legacy.post_process(old_raw, topk=5)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = ClassificationTask(runner, binding, top_k=5)
new_raw = task.forward(task.pre_process(image))
result = task.post_process(new_raw)
out = Path("outputs") / ("vit-" + variant + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
out.mkdir(parents=True, exist_ok=False)
np.save(out / "legacy.npy", old_raw[legacy.model_name][legacy.output_names[0]])
np.save(out / "unified.npy", new_raw[binding.output_name])
print("legacy", old_top)
print("unified", result.class_ids.tolist(), result.scores.tolist())
print("raw outputs:", out)
np.testing.assert_array_equal(result.class_ids, [i for i, _ in old_top])
np.testing.assert_allclose(result.scores, [s for _, s in old_top], rtol=0, atol=1e-5)
PYTHON
```

<a id="metrics"></a>
## Metrics

Host NV12 planes are byte-identical to source for two resize modes and three image shapes. Synthetic F32 logits give identical Top-5 IDs and scores within 1e-7. Proposed board criterion: identical IDs, absolute score error ≤1e-5; exact ties require per-ID evidence, no automatic relaxation. Top-1/Top-5 dataset accuracy measures whether the GT is among the selected classes; no new dataset result.

<a id="outputs"></a>
## Outputs

Tests print unittest results. Board recipe writes complete legacy.npy/unified.npy into a new outputs/vit-<variant>-<UTC> directory and prints both Top-K results. Separately retain board image/SDK identity, exact argv/cwd/UTC/rc/full stdout+stderr, code/deployed-file/model/image/labels digests and runtime metadata.

<a id="reference-results"></a>
## Reference results

Source: `rdk_s @380e1a2bf42041af54be6f34935e50197cfadff9`, `samples/vision/vit/evaluator/README.md`.

| Model | Top-1 | Top-5 |
| --- | --- | --- |
| ONNX | 74.54% | 98.36% |
| HBM | 72.62% | 98.03% |

Source states PTQ with 50 calibration images, no QAT. It does not separate int8 and int16 results or provide a complete benchmark receipt; values are historical, not measured here. Unified board and dataset results: not-run.

<a id="boundaries"></a>
## Boundaries

No dedicated dataset evaluator or latency benchmark is included. Host regressions cannot certify BPU behavior, current model bytes, int8/int16 accuracy or OE reproducibility. Raw output rejection is deliberate until an artifact-specific quantization contract is evidenced.
