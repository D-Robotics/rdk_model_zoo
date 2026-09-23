# RepGhost evaluation

<a id="dataset"></a>
## Dataset

Functional input: bundled `test_data/ibex.JPEG`. No dataset-level evaluator is delivered by the source; ImageNet validation data and its preparation are not included. One image cannot prove ImageNet accuracy.

<a id="environment"></a>
## Environment

Host: sample requirements (SciPy for source comparison). Board: matching X5 runtime environment described in the runtime README. The same classification task is exercised; this directory provides instructions, not a separate benchmark executable.

<a id="command"></a>
## Commands

```bash
# cwd: repository root; expected: unittest OK, exit 0
python3 -m unittest discover -s samples/vision/repghost/tests -v
```

Functional board run (duration not measured):

```bash
# cwd: repository root
bash samples/vision/repghost/model/download.sh x5 100
python3 samples/vision/repghost/runtime/python/main.py \
  --target x5 --variant 100 \
  --test-img samples/vision/repghost/test_data/ibex.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

Repeat for each published variant and both X5 board memory configurations. Use the preserved `repghost.py` task API for the baseline with identical artifact bytes, image, resize and Top-K. The source main.py calls an unimported `save_image` after inference, so its CLI exit code cannot be treated as a clean baseline success. Do not alter that source silently. Save complete output and the raw score vector before label formatting; the API example below bypasses only the broken visualization entrypoint.

```bash
# cwd: repository root on X5, prepare variant 100 first
PYTHONPATH="$PWD:$PWD/platforms/x5/samples/vision/repghost/runtime/python" python3 - <<'PYTHON'
import cv2
import numpy as np
from repghost import RepGhost, RepGhostConfig
from samples.vision.repghost.runtime.python.model_binding import resolve_selection
from samples.vision.repghost.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.repghost.runtime.python.classification import ClassificationTask

model_path = "samples/vision/repghost/model/RepGhost_100_224x224_nv12.bin"
image = cv2.imread("samples/vision/repghost/test_data/ibex.JPEG")
if image is None:
    raise FileNotFoundError("ibex.JPEG")
legacy = RepGhost(RepGhostConfig(model_path, resize_type=1, topk=5))
legacy.set_scheduling_params(priority=0, bpu_cores=[0])
old_outputs = legacy.forward(legacy.pre_process(image))
old_ids, old_scores, _ = legacy.post_process(old_outputs)
selection = resolve_selection("x5", variant="100")
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = ClassificationTask(runner, binding, top_k=5, resize_type=1)
new_outputs = task.forward(task.pre_process(image).tensors)
result = task.post_process(new_outputs)
# Preserve full vectors in a new output directory; do not overwrite old evidence.
from pathlib import Path
from datetime import datetime, timezone
out = Path("outputs") / ("repghost-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
out.mkdir(parents=True, exist_ok=False)
np.save(out / "legacy.npy", old_outputs[legacy.output_names[0]])
np.save(out / "unified.npy", new_outputs[binding.output_name])
print("legacy", old_ids.tolist(), old_scores.tolist())
print("unified", result.class_ids.tolist(), result.scores.tolist())
print("raw outputs:", out)
np.testing.assert_array_equal(result.class_ids, old_ids)
np.testing.assert_allclose(result.scores, old_scores, rtol=0, atol=1e-5)
PYTHON
```

This is an unexecuted board comparison recipe, not a receipt. A tie assertion failure requires inspecting per-ID scores rather than relaxing tolerances. Record the identities described under Outputs alongside these arrays.

<a id="metrics"></a>
## Metrics

Host preprocessing: exact bytes for three deterministic image shapes, both resize modes. Host score comparison: same synthetic F32 outputs, exact Top-K IDs and absolute score difference ≤1e-7. Proposed board criterion: identical IDs and abs(score difference)<1e-5; exact ties require per-ID evidence, not silent acceptance. Dataset accuracy and timing are not measured.

<a id="outputs"></a>
## Outputs

Host tests print unittest output; board CLI prints results and optionally a visualization. For board evidence record code SHA/deployed file hashes, board identity/SDK, exact argv/cwd, UTC times, return code, full stdout/stderr, artifact/image/labels hashes, metadata and raw outputs.

<a id="reference-results"></a>
## Reference results

Migration board comparison, dataset accuracy and timing: **not-run**. Historical table copied from the source evaluator at `rdk_x5 @ac115717197920355fc390bb04299b20e6436864`.

Source conditions: X5 CPU 8×A55@1.8GHz performance mode, BPU Bayes-e@1GHz. Float Top-1 is pre-quantization ONNX; Quant Top-1 is deployment. Single-thread latency is one frame/one BPU core; multi-thread latency and FPS use concurrent submissions. The source does not fix dataset subset, warm-up or repetition counts, so reproducibility remains incomplete.

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread Latency (ms) | Multi-thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RepGhost_200 | 224x224 | 9.79 | 76.43 | 75.25 | 2.89 | 8.76 | 451.42 |
| RepGhost_150 | 224x224 | 6.57 | 74.75 | 73.50 | 2.20 | 6.30 | 626.60 |
| RepGhost_130 | 224x224 | 5.48 | 75.00 | 73.57 | 1.87 | 5.30 | 743.56 |
| RepGhost_111 | 224x224 | 4.54 | 72.75 | 71.25 | 1.71 | 4.47 | 881.19 |
| RepGhost_100 | 224x224 | 4.07 | 72.50 | 72.25 | 1.55 | 4.08 | 964.69 |

<a id="boundaries"></a>
## Boundaries

Host tests neither execute the BPU nor certify compiler-generated artifacts. OE export/quantization, dataset accuracy, latency and stability remain untested. Published benchmark values are not new measurements.
