English | [简体中文](README_cn.md)

# MobileNetV3 evaluation

Use [`evaluate.py`](evaluate.py) to evaluate the FP32 ONNX exported from the selected timm checkpoint on a host. Follow the [shared host workflow](../../../../utils/tools/mobilenet/README.md) for installation and commands. Apply the checkpoint’s center-crop preprocessing and evaluate batch-one logits over the complete dataset.
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

Host FP32 ONNX evaluation uses `evaluate.py`, Python 3.10, and the dependencies in `utils/tools/mobilenet/requirements.lock.txt`. Follow the [shared host workflow](../../../../utils/tools/mobilenet/README.md) for installation and commands.

Board classification require the matching board image with `hbm_runtime`, NumPy, OpenCV-Python, PyYAML, the model, and labels. See the [Python runtime environment](../runtime/python/README.md#environment).

<a id="command"></a>
## Command

Board classification on X5 (prerequisite:
`bash samples/vision/mobilenetv3/model/download.sh x5`; output: exit 0 and the Top-K classes and scores):

```bash
python3 samples/vision/mobilenetv3/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv3:MobileNetV3_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv3/model/MobileNetV3_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv3/test_data/kit_fox.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

On S100/S600 substitute the `s:` reference and the `s100/`/`s600/`
artifact path; the labels file is shared. For a same-board comparison between runs, keep the compared run fixed —
same image, model bytes, labels, resize type, and Top-K — and compare
class IDs and raw scores before label formatting; expect identical IDs
and scores within 1e-5. Comparing X5 against S results is not a
same-board comparison.

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| Top-K agreement | identical post-softmax Top-K class IDs across repeated runs of the same artifact; scores within 1e-5 | same board, same artifact bytes, image, resize type, Top-K |
| Top-1 accuracy | fraction of argmax-correct predictions over the prepared ImageNet ILSVRC2012 validation set | same artifact, same resize type and Top-K as the single-image classification |
| latency / FPS | inference timing on the matching board | compare with the published figures under [Reference results](#reference-results), measured under the conditions stated there |

<a id="outputs"></a>
## Outputs

The board command prints Top-K class IDs, scores, and labels to stdout. Use `--img-save-path` to save an annotated image. For comparisons, record the board, model, command, input image, resize mode, and raw scores with the results.

<a id="reference-results"></a>
## Reference results

Figures published in the X5 release (x5-v1.1.3):

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Large | 224x224 | 1000 | 5.5 | 74.8% | 64.8% | 2.02 | 714+ |

For S100/S600 artifacts from release `s-v1.1.2`, run the selected artifact
on the matching board and use [Dataset-level evaluation](#boundaries) to
calculate accuracy and timing.

### RDK X5 / X5 Module performance

Data version: `rdk_x5_legacy @ cb86079ae5befcef9ca50fb46c8a6d8980106dec`.

X5 uses 1×Bayes-e. The eight-thread condition below applies to X3 measurements.

Published performance on RDK X5 and RDK X5 Module:


| Model        | Size    | Categories | Parameter | Floating point precision | Quantization accuracy | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ------------ | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------- |
| Mobilenetv3_large_100   | 224x224 | 1000 | 5.47   | 74.75 | 64.75 | 2.02        | 5.53        | 714.22 |


Description:
1. X5: CPU 8xA55@1.8G with all cores using Performance scheduling; BPU 1xBayes-e@1G, 10TOPS INT8.
2. Single-thread latency measures one frame on one BPU core.
3. Four-thread FPS uses four concurrent task submissions, with BPU utilization at 100%.
4. The X3 eight-thread measurement uses eight concurrent task submissions to its two BPU cores.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference accuracy Level of onnx before the model is quantized, while quantized accuracy is the accuracy Level of the actual inference of the model after quantization.

<a id="boundaries"></a>
## Dataset-level evaluation

For dataset Top-1 accuracy, pass each validation image to the runtime entry through `--test-img`, compare the returned Top-1 class ID with that image’s ground-truth model index, and divide correct predictions by the number of labeled images evaluated. Keep the artifact, resize mode, Top-K, board image and scheduling settings fixed when comparing runs. For latency or FPS, time the inference stage on the matching board and record the thread count and operating mode alongside the result.
