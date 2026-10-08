English | [简体中文](README_cn.md)

# EfficientViT evaluation
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

Host checks need the repository's user-space Python dependencies
(`requirements-host.txt` at the sample root) and no board SDK. The
functional board check needs an X5 board with its `hbm_runtime` image, a
prepared artifact, and the label file. The dataset-level evaluation would
additionally need the OE/board toolchain stated with the result.

<a id="command"></a>
## Command

Host checks (cwd: repository root; success: all tests OK, exit 0):

```bash
python3 -m unittest discover -s samples/vision/efficientvit/tests -v
```

Functional board check on X5 (prerequisite:
`bash samples/vision/efficientvit/model/download.sh x5`; success: exit 0
and a Top-5 containing hook-related classes):

```bash
python3 samples/vision/efficientvit/runtime/python/main.py \
  --target x5 \
  --asset-id x5:efficientvit:EfficientViT_m5_224x224_nv12.bin \
  --model-path samples/vision/efficientvit/model/EfficientViT_m5_224x224_nv12.bin \
  --test-img samples/vision/efficientvit/test_data/hook.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

For a same-board comparison between runs, keep the compared run fixed —
same image, model bytes, labels, resize type, and Top-K — and compare
class IDs and Top-K scores before label formatting; expect identical IDs
and scores within 1e-5. The output should be finite, non-zero, and stable
across repeated runs with the same input.

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| contract pass | runtime accepts the artifact, tensor names/shapes/dtypes match the binding, one F32 score vector returns | any prepared artifact on its matching board |
| Top-K agreement | identical post-softmax Top-K class IDs across repeated runs of the same artifact; scores within 1e-5 | same board, same artifact bytes, image, resize type, Top-K |
| Top-1 accuracy | fraction of argmax-correct predictions over the prepared ImageNet ILSVRC2012 validation set | same artifact, same resize type and Top-K as the functional check |
| latency / FPS | inference timing on the matching board | compare with the published figures under [Reference results](#reference-results), measured under the conditions stated there |

<a id="outputs"></a>
## Outputs

Host checks print the unittest result. The functional board check prints
the Top-K (class ids, scores, labels) on stdout and optionally writes an
annotated image with `--img-save-path`. For evidence, save the board
identity, model reference, runtime metadata, raw F32 score tensor, Top-K
output, image path, resize type, and command line.

<a id="reference-results"></a>
## Reference results

Figures published in the X5 release (x5-v1.1.3; Float Top-1 on the pre-quantization ONNX and Quant Top-1 on the deployment model). For new latency comparisons, record the thread count and concurrency settings:

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientViT_m5 | 224x224 | 1000 | 12.4 | 73.75% | 72.50% | 6.34 | 174.70 |

The published quantized Top-1 is 72.50% and Float Top-1 is 73.75%. Use the same dataset and preprocessing when comparing these results.

### RDK X5 performance

Data version: x5-v1.1.3. The `.bin` model uses `hbm_runtime`, CPU 8×A55@1.8GHz with full-core Performance scheduling, and BPU 1×Bayes-e@1GHz with 10TOPS INT8 compute. Single-thread latency measures one frame on one BPU core; multi-thread latency and FPS measure concurrent task submission.

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread Latency (ms) | Multi-thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientViT_m5 | 224x224 | 12.4 | 73.75% | 72.50% | 6.34 | 22.69 | 174.70 |

<a id="boundaries"></a>
## Dataset-level evaluation

For dataset Top-1 accuracy, pass each validation image to the runtime entry through `--test-img`, compare the returned Top-1 class ID with that image’s ground-truth model index, and divide correct predictions by the number of labeled images evaluated. Keep the artifact, resize mode, Top-K, board image and scheduling settings fixed when comparing runs. For latency or FPS, time the inference stage on the matching board and record the thread count and operating mode alongside the result.
