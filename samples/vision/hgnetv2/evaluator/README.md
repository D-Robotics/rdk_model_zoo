# HGNetV2 evaluation

<a id="dataset"></a>
## Dataset

Use a separately obtained ImageNet-1k validation set (normally 50,000 images); no dataset or download is bundled. The evaluator recursively scans JPEG/PNG files and matches CSV paths relative to --image-path, preserving subdirectories. Prepare CSV with UTF-8 header image:file,category and zero-based labels 0–999. Example layout:

```text
/data/imagenet-val/n01440764/example.JPEG
/data/imagenet-labels.csv:
image:file,category
n01440764/example.JPEG,0
```

Replace the example with real dataset labels. Missing, invalid or conflicting CSV categories are errors; backslashes in paths are normalized to slashes. Labels absent from the CSV are counted as unmatched, not inferred from directory names.

<a id="environment"></a>
## Environment

eval.py reuses the unified HGNetV2 ClassificationTask and RuntimeModelRunner on X5. It needs the same board SDK, NumPy, OpenCV and PyYAML; no additional inference framework is used. --help and pure dataset/metric tests run on the host without SDK. SciPy is only used by source-comparison host tests.

<a id="command"></a>
## Commands

```bash
# cwd: repository root; expected: unittest OK, exit 0
python3 -m unittest discover -s samples/vision/hgnetv2/tests -v
```

Functional board run (duration not measured):

```bash
# cwd: repository root
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/runtime/python/main.py \
  --target x5 --variant b0 \
  --test-img samples/vision/hgnetv2/test_data/sandbar.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

Repeat for each published variant and both X5 board memory configurations. Use the preserved source task API below for a same-process comparison. Keep model bytes, image, resize, Top-K and scheduling identical. The original CLI also remains available, but its formatted score output is less precise than the raw arrays saved here.

```bash
# cwd: repository root on X5, prepare variant b0 first
PYTHONPATH="$PWD:$PWD/platforms/x5/samples/vision/hgnetv2/runtime/python" python3 - <<'PYTHON'
import cv2
import numpy as np
from hgnetv2 import HGNetV2, HGNetV2Config
from samples.vision.hgnetv2.runtime.python.model_binding import resolve_selection
from samples.vision.hgnetv2.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.hgnetv2.runtime.python.classification import ClassificationTask

model_path = "samples/vision/hgnetv2/model/hgnetv2_b0_224x224_nv12.bin"
image = cv2.imread("samples/vision/hgnetv2/test_data/sandbar.JPEG")
if image is None:
    raise FileNotFoundError("sandbar.JPEG")
legacy = HGNetV2(HGNetV2Config(model_path, resize_type=1, topk=5))
legacy.set_scheduling_params(priority=0, bpu_cores=[0])
old_outputs = legacy.forward(legacy.pre_process(image))
old_ids, old_scores, _ = legacy.post_process(old_outputs)
selection = resolve_selection("x5", variant="b0")
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = ClassificationTask(runner, binding, top_k=5, resize_type=1)
new_outputs = task.forward(task.pre_process(image).tensors)
result = task.post_process(new_outputs)
# Preserve full vectors in a new output directory; do not overwrite old evidence.
from pathlib import Path
from datetime import datetime, timezone
out = Path("outputs") / ("hgnetv2-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
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

Dataset evaluation (cwd: repository root on X5). Download b0 first, replace data paths with your actual set. Duration depends on image count and is not measured here. Success returns 0 and writes the JSON; errors/no successful inference return 2. Unlabelled images can yield a partial report with exit 0, so check coverage fields.

```bash
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/evaluator/eval.py \
  --target x5 --variant b0 \
  --image-path /data/imagenet-val --val-csv /data/imagenet-labels.csv \
  --resize-type 0 --top-k 5 --limit 0 \
  --json-save-path outputs/hgnetv2-b0-evaluation.json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--target` | auto | Execution target; only matching X5 executes |
| `--variant` | null | b0 default; b1/b2/b3/b4 explicit |
| `--asset-id` | null | Exact manifest reference for external model path |
| `--model-path` | null | Prepared file; requires --asset-id, no automatic download |
| `--image-path` | required | Validation image root, recursive |
| `--val-csv` | required | Relative-path/category CSV |
| `--label-file` | empty string | Optional labels for display, not ground truth |
| `--json-save-path` | hgnetv2_cls_results.json | JSON output, overwrites named file |
| `--limit` | 0 | First N sorted images before label matching; 0=all |
| `--top-k / --topk` | 5 | Top-K accuracy, 1–1000 |
| `--resize-type` | 0 | Direct resize; 1=letterbox, differs from runtime default 1 |
| `--priority` | 0 | Scheduling priority |
| `--bpu-cores` | [0] | BPU core indices |

<a id="metrics"></a>
## Metrics

Top-1 = correct rank-1 / successful_inferences; Top-K = truth found in the K predictions / successful_inferences. The denominator preserves the source behavior, so missing/failed images must be considered alongside these rates. top5_acc is retained only when K=5; topk_acc is always the correctly named metric. FPS measures image reads plus preprocessing/inference/postprocessing in the loop, excludes model load/CSV/directory scan and includes no warm-up. It is not the source table’s multi-thread throughput. For fixed-image migration comparison, use identical IDs and absolute score difference <1e-5; exact ties need per-ID evidence.

<a id="outputs"></a>
## Outputs

Writes --json-save-path and prints the report. Fields include status (complete/partial/no-results), scanned/matched/unmatched/failed/successful counts, per-image errors, accuracy_denominator, top1_acc/topk_acc (null if no successful inference), optional top5_acc, elapsed_seconds/fps, asset_id/target/model, data paths and configuration. Complete means only all scanned images were evaluated; it does not certify a full 50,000-image dataset. Preserve stdout/stderr, code/deployed hashes, SDK/board identity and dataset/model hashes separately for reproducible evidence.

<a id="reference-results"></a>
## Reference results

Migration board comparison, dataset accuracy and timing: **not-run**. Historical table copied from the source evaluator at `rdk_x5 @ac115717197920355fc390bb04299b20e6436864`.

Source conditions: X5 CPU 8×A55@1.8GHz performance mode, BPU Bayes-e@1GHz. Float Top-1 is pre-quantization ONNX; Quant Top-1 is deployment. Single-thread latency is one frame/one BPU core; multi-thread latency and FPS use concurrent submissions. The source does not fix dataset subset, warm-up or repetition counts, so reproducibility remains incomplete.

| Model | Input Size | Params (M) | Float Top-1 | Quantized Top-1 | Single‑thread Latency (ms) | Multi‑thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HGNetv2_b0 | 224x224 | 6.0 | 77.342 | 72.17 | 1.96 | 3.29 | 902.09 |
| HGNetv2_b1 | 224x224 | 6.34 | 78.872 | 73.47 | 2.41 | 3.89 | 760.13 |
| HGNetv2_b2 | 224x224 | 11.2 | 81.578 | 75.55 | 3.52 | 7.41 | 401.16 |
| HGNetv2_b3 | 224x224 | 16.3 | 82.916 | 76.51 | 4.53 | 10.37 | 287.27 |
| HGNetv2_b4 | 224x224 | 19.8 | 83.694 | 81.93 | 5.29 | 12.32 | 241.94 |

<a id="boundaries"></a>
## Boundaries

The migrated evaluator has host tests only; no dataset or board evaluation was run. It does not compute ONNX accuracy, calibration quality, single-core latency or multi-thread throughput. Source CLI names remain where practical; external --model-path now requires --asset-id, invalid CSV is rejected instead of silently skipped, failures return nonzero, and topk_acc replaces the misleading top5 name for K other than 5.
