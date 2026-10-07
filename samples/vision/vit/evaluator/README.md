# ViT evaluation
Use the bundled image for a single-image classification check. For dataset accuracy, prepare the matching validation set and per-image ground-truth class indices, then compare those indices with the runtime’s Top-1 class IDs.

<a id="dataset"></a>

## Dataset
The functional check uses the ten bundled CIFAR-10 images, one per class. Dataset-level accuracy uses the complete CIFAR-10 test set. Prepare each test image with its ground-truth class index (0–9) and compare it with the runtime’s Top-1 class ID; the bundled examples provide one image for each class.

<a id="environment"></a>
## Environment

[Runtime prerequisites](../runtime/python/README.md#environment). The
functional board check needs an S100 board with `hbm_runtime` and the
prepared artifact; no OE environment is required for evaluation.

<a id="command"></a>
## Commands

```bash
# cwd: repository root; expected: unittest OK, exit 0
python3 -m unittest discover -s samples/vision/vit/tests -v
```

Prepare the model, then run the functional check on the matching board;
repeat with the int16 artifact and the other bundled images for broader
functional coverage:

```bash
# cwd: repository root
bash samples/vision/vit/model/download.sh s100 int8
python3 samples/vision/vit/runtime/python/main.py --target s100 --variant int8 \
  --test-img samples/vision/vit/test_data/airplane_0000.png \
  --label-file samples/vision/vit/test_data/cifar10_classes.names --top-k 5
```

For a same-board comparison between runs, keep the artifact bytes, image,
resize type and Top-K identical and compare class IDs and raw scores before
label formatting; expect identical IDs and scores within 1e-5. When a
Top-K boundary is an exact tie, compare per-ID scores instead of relaxing
the tolerance.

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| contract pass | runtime accepts the artifact, tensor names/shapes/dtypes match the binding, one F32 score vector returns | prepared artifact on a matching S100 board |
| Top-K agreement | identical post-softmax Top-K class IDs across repeated runs of the same artifact; scores within 1e-5 | same board, same artifact bytes, image, resize type, Top-K |
| Top-1 / Top-5 accuracy | whether the ground-truth class is the rank-1 / among the K selected classes | measured over the user-prepared CIFAR-10 test set |
| latency / FPS | inference timing on the matching board | measure with the runtime entry on the matching board |

<a id="outputs"></a>
## Outputs

Tests print unittest results. The functional board check prints the Top-K
(class ids, scores, labels) on stdout and optionally writes an annotated
image with `--img-save-path`. When recording a run, save the board
identity, model reference, command line, raw F32 score tensor and Top-K
output alongside the image path and resize type.

<a id="reference-results"></a>
## Reference results

Published in the original ViT evaluator record (CIFAR-10):

| Model | Top-1 | Top-5 |
| --- | --- | --- |
| ONNX | 74.54% | 98.36% |
| HBM | 72.62% | 98.03% |

The published record states PTQ with 50 calibration images, no QAT; the
int8 and int16 results are not separated in the record.

<a id="boundaries"></a>
## Dataset-level evaluation

For dataset Top-1 accuracy, pass each validation image to the runtime entry through `--test-img`, compare the returned Top-1 class ID with that image’s ground-truth model index, and divide correct predictions by the number of labeled images evaluated. Keep the artifact, resize mode, Top-K, board image and scheduling settings fixed when comparing runs. For latency or FPS, time the inference stage on the matching board and record the thread count and operating mode alongside the result.
