English | [简体中文](README_cn.md)

# HGNetV2 evaluation
This guide covers a one-image functional check and dataset-level Top-K evaluation on X5. The dataset command reads image paths and per-image ground-truth class indices from a CSV.

<a id="dataset"></a>

## Dataset

For ImageNet-1k dataset evaluation, prepare the validation JPEGs and a UTF-8 CSV with the header `image:file,category`. Each row pairs a path relative to `--image-path` with that image’s ground-truth model class index (0–999). The evaluator recursively scans JPEG/PNG files and preserves subdirectories when matching paths. Example layout:

```text
/data/imagenet-val/n01440764/example.JPEG
/data/imagenet-labels.csv:
image:file,category
n01440764/example.JPEG,0
```

Replace the example with real dataset labels. Missing, invalid or conflicting CSV categories are errors; backslashes in paths are normalized to slashes. Labels absent from the CSV are counted as unmatched, not inferred from directory names.

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── eval.py  # Python script
```

<a id="environment"></a>
## Environment

eval.py uses HGNetV2Classifier with the shared RuntimeModelRunner on X5. It needs the same board SDK, NumPy, OpenCV and PyYAML; no additional inference framework is used. --help and pure dataset/metric tests run on the host without SDK. SciPy is only used by source-comparison host tests.

<a id="command"></a>
## Commands

```bash
# cwd: repository root; expected: unittest OK, exit 0
python3 -m unittest discover -s samples/vision/hgnetv2/tests -v
```

Functional board run:

```bash
# cwd: repository root
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/runtime/python/main.py \
  --target x5 --variant b0 \
  --test-img samples/vision/hgnetv2/test_data/sandbar.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

Repeat for each published variant on the matching board. For a same-board
comparison between runs, keep model bytes, image, resize type, Top-K and
scheduling parameters identical and compare class IDs and raw scores before
label formatting; expect identical IDs and scores within 1e-5. When a
Top-K boundary is an exact tie, compare per-ID scores instead of relaxing
the tolerance.

Dataset evaluation (cwd: repository root on X5). Prepare the validation images and CSV, then run the command below. Runtime depends on image count. The command writes the JSON report; review its coverage counts together with the accuracy fields.

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

Top-1 = correct rank-1 / successful_inferences; Top-K = truth found in
the K predictions / successful_inferences. The denominators count only
successful inferences, so read the missing/failed counters alongside these
rates. `top5_acc` is written only when K=5; `topk_acc` is the correctly
named metric for every K. FPS measures image reads plus
preprocessing/inference/postprocessing in the loop, excludes model load
and CSV/directory scanning, and includes no warm-up — it is not the
multi-thread throughput of the published table. For fixed-image
comparisons use identical IDs and absolute score difference <1e-5; exact
ties need per-ID evidence.

<a id="outputs"></a>
## Outputs

Writes `--json-save-path` and prints the report. Fields include status (complete/partial/no-results), scanned/matched/unmatched/failed/successful counts, per-image errors, accuracy_denominator, top1_acc/topk_acc (null if no successful inference), optional top5_acc, elapsed_seconds/fps, asset_id/target/model, data paths and configuration. For a full ImageNet validation run, set `--limit 0` and check that 50,000 images are scanned, matched and successful, with zero unmatched or failed images. Keep the JSON report with the model and dataset identity used for the run.

<a id="reference-results"></a>
## Reference results

Figures published in the X5 release (x5-v1.1.3).

Conditions: X5 CPU 8×A55@1.8GHz performance mode, BPU Bayes-e@1GHz. Float
Top-1 is pre-quantization ONNX; Quant Top-1 is deployment. Single-thread
latency is one frame/one BPU core; multi-thread latency and FPS use
concurrent submissions. For new comparisons, use the same dataset subset and
record warm-up, repetition count, board mode and concurrency settings.

| Model | Input Size | Params (M) | Float Top-1 | Quantized Top-1 | Single‑thread Latency (ms) | Multi‑thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HGNetv2_b0 | 224x224 | 6.0 | 77.342 | 72.17 | 1.96 | 3.29 | 902.09 |
| HGNetv2_b1 | 224x224 | 6.34 | 78.872 | 73.47 | 2.41 | 3.89 | 760.13 |
| HGNetv2_b2 | 224x224 | 11.2 | 81.578 | 75.55 | 3.52 | 7.41 | 401.16 |
| HGNetv2_b3 | 224x224 | 16.3 | 82.916 | 76.51 | 4.53 | 10.37 | 287.27 |
| HGNetv2_b4 | 224x224 | 19.8 | 83.694 | 81.93 | 5.29 | 12.32 | 241.94 |

<a id="boundaries"></a>
## Dataset-level evaluation

Use the dataset command with `--limit 0` and all 50,000 validation image paths to evaluate the complete split. Review `scanned`, `matched`, `unmatched`, `failed`, and `successful` counts with the Top-K rates; a full-set run has 50,000 successful, labeled images. Pair an external `--model-path` with its exact `--asset-id`, provide valid CSV categories, and select the result field for the requested K (`top5_acc` for K=5, `topk_acc` for other values).
