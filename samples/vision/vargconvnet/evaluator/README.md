English | [简体中文](README_cn.md)

# VargConvNet evaluation
Use the bundled image for a single-image classification. For dataset accuracy, prepare the matching validation set and per-image ground-truth class indices, then compare those indices with the runtime’s Top-1 class IDs.

<a id="dataset"></a>

## Dataset
The single-image classification uses the bundled test image. Dataset-level accuracy uses ImageNet ILSVRC2012 validation (50,000 images, 1,000 classes). Prepare a ground-truth mapping from each image to its zero-based model class index and compare it with the runtime’s Top-1 class ID. `datasets/imagenet/imagenet_classes.names` maps output indices to display names; per-image truth comes from the dataset annotations. See [ImageNet preparation](../../../../datasets/imagenet/README.md).

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="environment"></a>
## Environment

Run on X5 with the environment described in the runtime README. Use the classification command below to collect results for evaluation.

<a id="command"></a>
## Commands


Board classification on the matching board:

```bash
# cwd: repository root
bash samples/vision/vargconvnet/model/download.sh x5 vargconvnet
python3 samples/vision/vargconvnet/runtime/python/main.py \
  --target x5 --variant vargconvnet \
  --test-img samples/vision/vargconvnet/test_data/box_turtle.JPEG \
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
| Top-K agreement | identical post-softmax Top-K class IDs across repeated runs of the same artifact; scores within 1e-5 | same board, same artifact bytes, image, resize type, Top-K |
| Top-1 accuracy | fraction of argmax-correct predictions over the prepared ImageNet ILSVRC2012 validation set | same artifact, same resize type and Top-K as the single-image classification |
| latency / FPS | inference timing on the matching board | compare with the published figures under [Reference results](#reference-results), measured under the conditions stated there |

<a id="outputs"></a>
## Outputs

Repeat the single-image classification on the matching board and compare with the published figures under [Reference results](#reference-results), using the measurement conditions stated there.

<a id="reference-results"></a>
## Reference results

For dataset Top-1 accuracy, compare each returned class ID with the ImageNet ground-truth mapping and divide correct predictions by labeled images evaluated. Measure latency on the matching board with the same preprocessing and concurrency settings used for the comparison.

<a id="boundaries"></a>
## Dataset-level evaluation

For dataset Top-1 accuracy, pass each validation image to the runtime entry through `--test-img`, compare the returned Top-1 class ID with that image’s ground-truth model index, and divide correct predictions by the number of labeled images evaluated. Keep the artifact, resize mode, Top-K, board image and scheduling settings fixed when comparing runs. For latency or FPS, time the inference stage on the matching board and record the thread count and operating mode alongside the result.
