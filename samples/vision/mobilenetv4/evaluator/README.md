English | [简体中文](README_cn.md)

# MobileNetV4 evaluation

Use [`evaluate.py`](evaluate.py) to evaluate the FP32 ONNX exported from the selected timm checkpoint on a host. Follow the [shared host workflow](../../../../utils/tools/mobilenet/README.md) for installation and commands. Apply the checkpoint’s center-crop preprocessing and evaluate batch-one logits over the complete dataset.
Use the bundled image for a single-image classification. For dataset accuracy, prepare the matching validation set and per-image ground-truth class indices, then compare those indices with the runtime’s Top-1 class IDs.


## Pinned board evaluation

`evaluate_board.py` loads one compiled model and evaluates every image in the
ImageNetV2 MatchedFrequency manifest. Use the matching artifact for the
board (X5 `.bin`; S100, S100P or S600 `.hbm`), with NV12 `bt601_video` input
and float32 logits, and select the checkpoint's preprocessing settings with
`--variant` (`v4-small` or `v4-medium-224` or `v4-large-256`). The evaluator uses PIL bicubic
shorter-edge resize (256 for small, 235 for medium, 269 for large) and a center crop to the model input (224 / 256),
followed by the sample's OpenCV NV12 conversion; this is the same geometry
the runtime applies with `--resize-type 2`. The required campaign binds
labels, geometry and the expected 10,000 images; every image and model hash is
checked.

```bash
python3 samples/vision/mobilenetv4/evaluator/evaluate_board.py \
  --target s100 --variant v4-small \
  --model samples/vision/mobilenetv4/model/s100/mobilenetv4_conv_small_nashe_224x224_nv12.hbm \
  --model-sha256 <sha256-from-model/README.md> \
  --campaign /path/to/campaign.json --manifest /path/to/manifest.json \
  --data-root /path/to/imagenetv2/images --output /path/to/new-evaluation
```

The new output directory contains full predictions, tensor metadata, three
fixed-input outputs and `evaluation.json` with Top-1/Top-5 and source hashes.
For C++ performance timing and its image boundary, see
[the benchmark instructions](../../../../utils/tools/mobilenet/cpp/README.md).

<a id="dataset"></a>

## Dataset
The single-image classification uses the bundled test image. The `evaluate_board.py` command above uses ImageNetV2 MatchedFrequency (10,000 images). The per-image procedure below uses ImageNet ILSVRC2012 validation (50,000 images, 1,000 classes). Prepare a ground-truth mapping from each image to its zero-based model class index and compare it with the runtime’s Top-1 class ID. `datasets/imagenet/imagenet_classes.names` maps output indices to display names; per-image truth comes from the dataset annotations. See [ImageNet preparation](../../../../datasets/imagenet/README.md).

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── evaluate.py  # FP32 ONNX evaluation (shared host workflow)
└── evaluate_board.py  # Board evaluation of a compiled artifact
```

<a id="environment"></a>
## Environment

Host FP32 ONNX evaluation uses `evaluate.py`, Python 3.10, and the dependencies in `utils/tools/mobilenet/requirements.lock.txt`. Follow the [shared host workflow](../../../../utils/tools/mobilenet/README.md) for installation and commands.

Board classification and `evaluate_board.py` require the matching board image with `hbm_runtime`, NumPy, OpenCV-Python, Pillow, PyYAML, the model, and labels. See the [Python runtime environment](../runtime/python/README.md#environment).

<a id="command"></a>
## Command

Board classification on X5 (prerequisite:
`bash samples/vision/mobilenetv4/model/download.sh x5 small`; output: exit 0 and the Top-K classes and scores):

```bash
python3 samples/vision/mobilenetv4/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv4:mobilenetv4_conv_small_bayese_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv4/model/mobilenetv4_conv_small_bayese_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv4/test_data/great_grey_owl.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

On S100, S100P and S600 substitute the `s:` reference and the `s100/`,
`s100p/` or `s600/` artifact path; the labels file is shared. For a same-board comparison between runs, keep the compared run fixed —
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

Accuracy of the published models, measured with the pinned board evaluation
above on the complete ImageNetV2 MatchedFrequency set (10,000 images). This is
not the ILSVRC2012 validation set, so the values are not comparable with
ImageNet-1k validation figures. "FP32" is the ONNX export of the same
checkpoint on the same crops.

| Model | Target | FP32 Top-1 | Board Top-1 | FP32 Top-5 | Board Top-5 |
| --- | --- | --- | --- | --- | --- |
| MobileNetV4-Conv-Small | X5 | 60.96% | 58.72% | 82.64% | 80.53% |
| MobileNetV4-Conv-Small | S100 | 60.96% | 58.56% | 82.64% | 80.78% |
| MobileNetV4-Conv-Small | S100P | 60.96% | 58.56% | 82.64% | 80.78% |
| MobileNetV4-Conv-Small | S600 | 60.96% | 58.57% | 82.64% | 80.94% |
| MobileNetV4-Conv-Medium | X5 | 67.35% | 66.25% | 87.86% | 87.54% |
| MobileNetV4-Conv-Medium | S100 | 67.35% | 66.94% | 87.86% | 87.65% |
| MobileNetV4-Conv-Medium | S100P | 67.35% | 66.94% | 87.86% | 87.65% |
| MobileNetV4-Conv-Medium | S600 | 67.35% | 66.75% | 87.86% | 87.56% |
| MobileNetV4-Conv-Large | X5 | 70.79% | 69.85% | 89.15% | 89.13% |
| MobileNetV4-Conv-Large | S100 | 70.79% | 69.66% | 89.15% | 89.04% |
| MobileNetV4-Conv-Large | S100P | 70.79% | 69.66% | 89.15% | 89.04% |
| MobileNetV4-Conv-Large | S600 | 70.79% | 69.62% | 89.15% | 89.16% |

Latency and throughput of the same artifacts, with the measurement
conditions, are in the [sample README](../README.md#performance). The X3
figures below use X3 builds with their own weights and preprocessing.

### RDK X3 / X3 Module performance

Data version: `rdk_x3 @ 0eb344ba8bed76923a6bd696e468fd82489cf46e`.

This table describes X3 hardware and its toolchain. See the sample support matrix for boards accepted by the current entry.

The following table shows the performance data obtained from actual testing on RDK X3 & RDK X3 Module.

| Model        | Size    | Categories | Parameter | Floating point precision | Quantization accuracy | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ------------ | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------- |
| Mobilenetv4 | 224x224 | 1000 | 3.76   | 70.50 | 70.26 | 1.43        | 2.96        | 1309.17 |


Description:
1. X3 is in the best state: CPU is 4xA53@1.5G, full core Performance scheduling, BPU is 2xBernoulli@1G, a total of 5TOPS equivalent int8 computing power.
2. Single-thread latency measures one frame on one BPU core.
3. Four-thread FPS uses four concurrent task submissions, with BPU utilization at 100%.
4. The X3 eight-thread measurement uses eight concurrent task submissions to its two BPU cores.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference accuracy Level of onnx before the model is quantized, while quantized accuracy is the accuracy Level of the actual inference of the model after quantization.

<a id="boundaries"></a>
## Dataset-level evaluation

For dataset Top-1 accuracy, pass each validation image to the runtime entry through `--test-img`, compare the returned Top-1 class ID with that image’s ground-truth model index, and divide correct predictions by the number of labeled images evaluated. Keep the artifact, resize mode, Top-K, board image and scheduling settings fixed when comparing runs. For latency or FPS, time the inference stage on the matching board and record the thread count and operating mode alongside the result.
