English | [简体中文](README_cn.md)

# ResNet evaluation

Use [`evaluate.py`](evaluate.py) to evaluate the FP32 ONNX exported from the selected timm checkpoint on a host. Follow the [shared host workflow](../../../../utils/tools/classification/README.md) for installation and commands. Apply the checkpoint’s center-crop preprocessing and evaluate batch-one logits over the complete dataset.
Use the bundled image for a single-image classification. For dataset accuracy, prepare the matching validation set and per-image ground-truth class indices, then compare those indices with the runtime’s Top-1 class IDs.


## Pinned board evaluation

`evaluate_board.py` loads one compiled model and evaluates every image in the
ImageNetV2 MatchedFrequency manifest. Use the matching artifact for the
board (X5 `.bin`; S100, S100P or S600 `.hbm`), with NV12 `bt601_video` input
and float32 logits, and select the checkpoint's preprocessing settings with
`--variant` (`resnet18` or `resnet34` or `resnet50` or `resnet101` or `resnet152`). The evaluator uses PIL bicubic
shorter-edge resize (235 for every variant) and a center crop to the model input (224),
followed by the sample's OpenCV NV12 conversion; this is the same geometry
the runtime applies with `--resize-type 2`. The required campaign binds
labels, geometry and the expected 10,000 images; every image and model hash is
checked.

```bash
python3 samples/vision/resnet/evaluator/evaluate_board.py \
  --target s100 --variant resnet18 \
  --model samples/vision/resnet/model/s100/resnet18_nashe_224x224_nv12.hbm \
  --model-sha256 <sha256-from-model/README.md> \
  --campaign /path/to/campaign.json --manifest /path/to/manifest.json \
  --data-root /path/to/imagenetv2/images --output /path/to/new-evaluation
```

The new output directory contains full predictions, tensor metadata, three
fixed-input outputs and `evaluation.json` with Top-1/Top-5 and source hashes.
For C++ performance timing and its image boundary, see
[the benchmark instructions](../../../../utils/tools/classification/cpp/README.md).

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

Host FP32 ONNX evaluation uses `evaluate.py`, Python 3.10, and the dependencies in `utils/tools/classification/requirements.lock.txt`. Follow the [shared host workflow](../../../../utils/tools/classification/README.md) for installation and commands.

Board classification and `evaluate_board.py` require the matching board image with `hbm_runtime`, NumPy, OpenCV-Python, Pillow, PyYAML, the model, and labels. See the [Python runtime environment](../runtime/python/README.md#environment).

<a id="command"></a>
## Command

Board classification on X5 (prerequisite:
`bash samples/vision/resnet/model/download.sh x5 18`; output: exit 0 and the Top-K classes and scores):

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_bayese_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_bayese_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
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
| ResNet-18 | X5 | 59.35% | 58.91% | 80.94% | 80.56% |
| ResNet-18 | S100 | 59.35% | 58.79% | 80.94% | 80.46% |
| ResNet-18 | S100P | 59.35% | 58.79% | 80.94% | 80.46% |
| ResNet-18 | S600 | 59.35% | 58.88% | 80.94% | 80.58% |
| ResNet-34 | X5 | 64.92% | 64.30% | 85.06% | 84.71% |
| ResNet-34 | S100 | 64.92% | 64.26% | 85.06% | 84.75% |
| ResNet-34 | S100P | 64.92% | 64.26% | 85.06% | 84.75% |
| ResNet-34 | S600 | 64.92% | 64.23% | 85.06% | 84.75% |
| ResNet-50 | X5 | 68.84% | 66.90% | 87.59% | 86.25% |
| ResNet-50 | S100 | 68.84% | 67.01% | 87.59% | 86.40% |
| ResNet-50 | S100P | 68.84% | 67.01% | 87.59% | 86.40% |
| ResNet-50 | S600 | 68.84% | 66.94% | 87.59% | 86.41% |
| ResNet-101 | X5 | 70.32% | 68.59% | 88.35% | 87.66% |
| ResNet-101 | S100 | 70.32% | 68.64% | 88.35% | 87.67% |
| ResNet-101 | S100P | 70.32% | 68.64% | 88.35% | 87.67% |
| ResNet-101 | S600 | 70.32% | 68.65% | 88.35% | 87.56% |
| ResNet-152 | X5 | 70.87% | 69.34% | 88.16% | 87.78% |
| ResNet-152 | S100 | 70.87% | 69.27% | 88.16% | 87.74% |
| ResNet-152 | S100P | 70.87% | 69.27% | 88.16% | 87.74% |
| ResNet-152 | S600 | 70.87% | 69.22% | 88.16% | 87.74% |

Latency and throughput of the same artifacts, with the measurement
conditions, are in the [sample README](../README.md#performance).
The X5 and X3 figures below belong to earlier builds (different weights and
preprocessing); compare them under the conditions shown with each table.

### RDK X5 / X5 Module performance

Data version: `rdk_x5_legacy @ cb86079ae5befcef9ca50fb46c8a6d8980106dec`.

X5 uses 1×Bayes-e. The eight-thread condition below applies to X3 measurements.

The following table shows the performance data obtained from actual testing on RDK X5 & RDK X5 Module.

| Model       | Size    | Categories | Parameter | Floating point precision | Quantization accuracy | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ----------- | ------- | ---------- | --------- | ------------------------ | --------------------- | ------------------------------------ | ----------------------------------- | --------------- |
| ResNet18 | 224x224 | 1000 | 11.2    | 71.49 | 70.50 | 2.95        | 8.81        | 448.79 |

Description:
1. X5: CPU 8xA55@1.8G with all cores using Performance scheduling; BPU 1xBayes-e@1G, 10TOPS INT8.
2. Single-thread latency measures one frame on one BPU core.
3. Four-thread FPS uses four concurrent task submissions, with BPU utilization at 100%.
4. The X3 eight-thread measurement uses eight concurrent task submissions to its two BPU cores.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference accuracy Level of onnx before the model is quantized, while quantized accuracy is the accuracy Level of the actual inference of the model after quantization.

### RDK X3 / X3 Module performance

Data version: `rdk_x3 @ 0eb344ba8bed76923a6bd696e468fd82489cf46e`.

This table describes X3 hardware and its toolchain. See the sample support matrix for boards accepted by the current entry.

The following table shows the performance data obtained from actual testing on RDK X3 & RDK X3 Module.

| Model       | Size    | Categories | Parameter | Floating point precision | Quantization accuracy | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ----------- | ------- | ---------- | --------- | ------------------------ | --------------------- | ------------------------------------ | ----------------------------------- | --------------- |
| ResNet18 | 224x224 | 1000 | 11.2    | 71.49 | 70.50 | 8.87        | 17.07        | 232.74 |

Description:
1. X3 is in the best state: CPU is 4xA53@1.5G, full core Performance scheduling, BPU is 2xBernoulli@1G, a total of 5TOPS equivalent int8 computing power.
2. Single-thread latency measures one frame on one BPU core.
3. Four-thread FPS uses four concurrent task submissions, with BPU utilization at 100%.
4. The X3 eight-thread measurement uses eight concurrent task submissions to its two BPU cores.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference accuracy Level of onnx before the model is quantized, while quantized accuracy is the accuracy Level of the actual inference of the model after quantization.

<a id="boundaries"></a>
## Dataset-level evaluation

For dataset Top-1 accuracy, pass each validation image to the runtime entry through `--test-img`, compare the returned Top-1 class ID with that image’s ground-truth model index, and divide correct predictions by the number of labeled images evaluated. Keep the artifact, resize mode, Top-K, board image and scheduling settings fixed when comparing runs. For latency or FPS, time the inference stage on the matching board and record the thread count and operating mode alongside the result.
