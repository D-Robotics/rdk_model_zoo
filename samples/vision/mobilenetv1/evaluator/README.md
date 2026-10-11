English | [简体中文](README_cn.md)

# MobileNetV1 evaluation

For the pinned timm checkpoint workflow, use [`evaluate.py`](evaluate.py) and the [shared host workflow](../../../../utils/tools/mobilenet/README.md). It records the exact checkpoint, center-crop preprocessing, batch-one logits contract, and full-dataset evaluation inputs. The existing artifact commands below retain their own contracts.
Use the bundled image for a single-image classification check. For dataset accuracy, prepare the matching validation set and per-image ground-truth class indices, then compare those indices with the runtime’s Top-1 class IDs.


## Pinned board evaluation

`evaluate_board.py` loads one compiled model and evaluates every image in the
frozen ImageNetV2 MatchedFrequency manifest. Use the matching artifact for the
board (X5 `.bin`; S100, S100P or S600 `.hbm`), with NV12 `bt601_video` input
and float32 logits, and select the checkpoint's preprocessing contract with
`--variant` (`v1-100` or `v1-125`). The evaluator uses PIL bicubic
shorter-edge resize (256 for 100, 248 for 125) and a center crop to the model input (224),
followed by the sample's OpenCV NV12 conversion; this is the same geometry
the runtime applies with `--resize-type 2`. The required campaign binds
labels, geometry and the expected 10,000 images; every image and model hash is
checked.

```bash
python3 samples/vision/mobilenetv1/evaluator/evaluate_board.py \
  --target s100 --variant v1-100 \
  --model samples/vision/mobilenetv1/model/s100/mobilenetv1_100_nashe_224x224_nv12.hbm \
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
The functional check uses the bundled test image. Dataset-level accuracy uses ImageNet ILSVRC2012 validation (50,000 images, 1,000 classes). Prepare a ground-truth mapping from each image to its zero-based model class index and compare it with the runtime’s Top-1 class ID. `datasets/imagenet/imagenet_classes.names` maps output indices to display names; per-image truth comes from the dataset annotations. See [ImageNet preparation](../../../../datasets/imagenet/README.md).

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

Host checks need the repository's user-space Python dependencies
(`requirements-host.txt` at the sample root) and no board SDK. The
functional board check needs the target board with its `hbm_runtime`
image, a prepared artifact, and the label file. The dataset-level
evaluation would additionally need the OE/board toolchain stated with the
result.

<a id="command"></a>
## Command

Host checks (cwd: repository root; success: all tests OK, exit 0):

```bash
python3 -m unittest discover -s samples/vision/mobilenetv1/tests -v
```

Functional board check on X5 (prerequisite:
`bash samples/vision/mobilenetv1/model/download.sh x5 100`; success: exit 0 and the expected Top-K):

```bash
python3 samples/vision/mobilenetv1/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv1:mobilenetv1_100_bayese_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv1/model/mobilenetv1_100_bayese_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv1/test_data/bulbul.JPEG \
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

Accuracy of the published models, measured with the pinned board evaluation
above on the complete ImageNetV2 MatchedFrequency set (10,000 images). This is
not the ILSVRC2012 validation set, so the values are not comparable with
ImageNet-1k validation figures. "FP32" is the ONNX export of the same
checkpoint on the same crops.

| Model | Target | FP32 Top-1 | Board Top-1 | FP32 Top-5 | Board Top-5 |
| --- | --- | --- | --- | --- | --- |
| MobileNetV1-100 | X5 | 62.86% | 59.45%* | 84.37% | 81.53% |
| MobileNetV1-100 | S100 | 62.86% | 62.00% | 84.37% | 83.51% |
| MobileNetV1-100 | S100P | 62.86% | 62.00% | 84.37% | 83.51% |
| MobileNetV1-100 | S600 | 62.86% | 62.04% | 84.37% | 83.66% |
| MobileNetV1-125 | X5 | 64.25% | 63.14% | 85.23% | 84.17% |
| MobileNetV1-125 | S100 | 64.25% | 63.19% | 85.23% | 84.27% |
| MobileNetV1-125 | S100P | 64.25% | 63.19% | 85.23% | 84.27% |
| MobileNetV1-125 | S600 | 64.25% | 62.95% | 85.23% | 84.12% |

\* Documented exception: MobileNetV1-100 on X5 loses 5.4% of its FP32 Top-1
(relative), above the 5% acceptance target; see the [sample README](../README.md#performance).

Latency and throughput of the same artifacts, with the measurement
conditions, are in the [sample README](../README.md#performance). 
The X5 and X3 figures below belong to earlier builds (different weights and
preprocessing) and are kept as history.

### RDK X5 / X5 Module performance

Data version: `rdk_x5_legacy @ cb86079ae5befcef9ca50fb46c8a6d8980106dec`.

The threading descriptions below are the conditions stated with these measurements. The dual-core and X3 eight-thread descriptions refer to X3; X5 has 1×Bayes-e.

The following table shows the performance data obtained from actual testing on RDK X5 & RDK X5 Module.

| Model       | Size    | Categories | Parameter | Floating point precision | Quantization accuracy | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ----------- | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------ |
| MobileNetv1 | 224x224 | 1000 | 1.33   | 71.74 | 65.36 | 1.27        | 2.90        | 1356.25 |

Description:
1. X5 is in the best state: CPU is 8xA55@1.8G, full core Performance scheduling, BPU is 1xBayes-e@1G, a total of 10TOPS equivalent int8 computing power.
2. Single-threaded delay is the ideal situation for single frame, single-threaded, and single-BPU core delay, and BPU inference for a task.
3. The frame rate of a 4-thread project is when 4 threads simultaneously send tasks to a dual-core BPU. In a typical project, 4 threads can control the single frame delay to be small, while consuming all BPUs to 100%, achieving a good balance between throughput (FPS) and frame delay.
4. The maximum frame rate of 8 threads is for 8 threads to simultaneously load tasks into the dual-core BPU of X3. The purpose is to test the maximum performance of the BPU. Generally, 4 cores are already full. If 8 threads are much better than 4 threads, it indicates that the model structure needs to improve the "calculation/memory access" ratio or optimize the DDR bandwidth when compiling.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference accuracy Level of onnx before the model is quantized, while quantized accuracy is the accuracy Level of the actual inference of the model after quantization.

### RDK X3 / X3 Module performance

Data version: `rdk_x3 @ 0eb344ba8bed76923a6bd696e468fd82489cf46e`.

This table describes X3 hardware and its toolchain. See the sample support matrix for boards accepted by the current entry.

The following table shows the performance data obtained from actual testing on RDK X3 & RDK X3 Module.

| Model       | Size    | Categories | Parameter | Floating point precision | Quantization accuracy | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ----------- | ------- | ---- | ------ | ----- | ----- | ----------- | ----------- | ------ |
| MobileNetv1 | 224x224 | 1000 | 1.33   | 71.74 | 65.36 | 3.44        | 6.10        | 647.83 |

Description:
1. X3 is in the best state: CPU is 4xA53@1.5G, full core Performance scheduling, BPU is 2xBernoulli@1G, a total of 5TOPS equivalent int8 computing power.
2. Single-threaded delay is the ideal situation for single frame, single-threaded, and single-BPU core delay, and BPU inference for a task.
3. The frame rate of a 4-thread project is when 4 threads simultaneously send tasks to a dual-core BPU. In a typical project, 4 threads can control the single frame delay to be small, while consuming all BPUs to 100%, achieving a good balance between throughput (FPS) and frame delay.
4. The maximum frame rate of 8 threads is for 8 threads to simultaneously load tasks into the dual-core BPU of X3. The purpose is to test the maximum performance of the BPU. Generally, 4 cores are already full. If 8 threads are much better than 4 threads, it indicates that the model structure needs to improve the "calculation/memory access" ratio or optimize the DDR bandwidth when compiling.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference accuracy Level of onnx before the model is quantized, while quantized accuracy is the accuracy Level of the actual inference of the model after quantization.

<a id="boundaries"></a>
## Dataset-level evaluation

For dataset Top-1 accuracy, pass each validation image to the runtime entry through `--test-img`, compare the returned Top-1 class ID with that image’s ground-truth model index, and divide correct predictions by the number of labeled images evaluated. Keep the artifact, resize mode, Top-K, board image and scheduling settings fixed when comparing runs. For latency or FPS, time the inference stage on the matching board and record the thread count and operating mode alongside the result.
