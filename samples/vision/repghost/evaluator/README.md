# RepGhost evaluation
Use the bundled image for a single-image classification check. For dataset accuracy, prepare the matching validation set and per-image ground-truth class indices, then compare those indices with the runtime’s Top-1 class IDs.

<a id="dataset"></a>

## Dataset
The functional check uses the bundled test image. Dataset-level accuracy uses ImageNet ILSVRC2012 validation (50,000 images, 1,000 classes). Prepare a ground-truth mapping from each image to its zero-based model class index and compare it with the runtime’s Top-1 class ID. `datasets/imagenet/imagenet_classes.names` maps output indices to display names; per-image truth comes from the dataset annotations. See [ImageNet preparation](../../../../datasets/imagenet/README.md).

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="environment"></a>
## Environment

Host: sample requirements (SciPy for source comparison). Board: matching X5 runtime environment described in the runtime README. Use the classification runtime below to collect results for evaluation.

<a id="command"></a>
## Commands

```bash
# cwd: repository root; expected: unittest OK, exit 0
python3 -m unittest discover -s samples/vision/repghost/tests -v
```

Functional board check on the matching board:

```bash
# cwd: repository root
bash samples/vision/repghost/model/download.sh x5 100
python3 samples/vision/repghost/runtime/python/main.py \
  --target x5 --variant 100 \
  --test-img samples/vision/repghost/test_data/ibex.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

Repeat for each published variant on the matching board. For a same-board
comparison between runs, keep model bytes, image, resize type, Top-K and
scheduling parameters identical and compare class IDs and raw scores before
label formatting; expect identical IDs and scores within 1e-5. When a
Top-K boundary is an exact tie, compare per-ID scores instead of relaxing
the tolerance.

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| contract pass | runtime accepts the artifact, tensor names/shapes/dtypes match the binding, one F32 score vector returns | prepared artifact on a matching X5 board |
| Top-K agreement | identical post-softmax Top-K class IDs across repeated runs of the same artifact; scores within 1e-5 | same board, same artifact bytes, image, resize type, Top-K |
| Top-1 accuracy | fraction of argmax-correct predictions over the prepared ImageNet ILSVRC2012 validation set | same artifact, same resize type and Top-K as the functional check |
| latency / FPS | inference timing on the matching board | compare with the published figures under [Reference results](#reference-results), measured under the conditions stated there |

<a id="outputs"></a>
## Outputs

Repeat the functional check on the matching board and compare with the published figures under [Reference results](#reference-results), using the measurement conditions stated there.

<a id="reference-results"></a>
## Reference results

The table below is from the published source evaluator.

Source conditions: X5 CPU 8×A55@1.8GHz performance mode, BPU Bayes-e@1GHz. Float Top-1 is pre-quantization ONNX; Quant Top-1 is deployment. Single-thread latency is one frame/one BPU core; multi-thread latency and FPS use concurrent submissions. For a new comparison, use the same dataset subset and record warm-up, repetition count, board mode and concurrency settings.

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread Latency (ms) | Multi-thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RepGhost_200 | 224x224 | 9.79 | 76.43 | 75.25 | 2.89 | 8.76 | 451.42 |
| RepGhost_150 | 224x224 | 6.57 | 74.75 | 73.50 | 2.20 | 6.30 | 626.60 |
| RepGhost_130 | 224x224 | 5.48 | 75.00 | 73.57 | 1.87 | 5.30 | 743.56 |
| RepGhost_111 | 224x224 | 4.54 | 72.75 | 71.25 | 1.71 | 4.47 | 881.19 |
| RepGhost_100 | 224x224 | 4.07 | 72.50 | 72.25 | 1.55 | 4.08 | 964.69 |

<a id="boundaries"></a>
## Dataset-level evaluation

For dataset Top-1 accuracy, pass each validation image to the runtime entry through `--test-img`, compare the returned Top-1 class ID with that image’s ground-truth model index, and divide correct predictions by the number of labeled images evaluated. Keep the artifact, resize mode, Top-K, board image and scheduling settings fixed when comparing runs. For latency or FPS, time the inference stage on the matching board and record the thread count and operating mode alongside the result.
