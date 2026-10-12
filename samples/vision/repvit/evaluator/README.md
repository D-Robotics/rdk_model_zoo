English | [简体中文](README_cn.md)

# RepViT evaluation
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
bash samples/vision/repvit/model/download.sh x5 m0_9
python3 samples/vision/repvit/runtime/python/main.py \
  --target x5 --variant m0_9 \
  --test-img samples/vision/repvit/test_data/yurt.JPEG \
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

The table below is from the published source evaluator.

Source conditions: X5 CPU 8×A55@1.8GHz performance mode, BPU Bayes-e@1GHz. Float Top-1 is pre-quantization ONNX; Quant Top-1 is deployment. Single-thread latency is one frame/one BPU core; multi-thread latency and FPS use concurrent submissions. For a new comparison, use the same dataset subset and record warm-up, repetition count, board mode and concurrency settings.

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread Latency (ms) | Multi-thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RepViT-m1.1 | 224x224 | 8.2 | 77.73% | 77.50% | 2.32 | 6.69 | 590.42 |
| RepViT-m1.0 | 224x224 | 6.8 | 76.75% | 76.50% | 1.97 | 5.71 | 692.29 |
| RepViT-m0.9 | 224x224 | 5.1 | 76.32% | 75.75% | 1.65 | 4.37 | 902.69 |

### RDK X5 / X5 Module performance

Data version: `rdk_x5_legacy @ cb86079ae5befcef9ca50fb46c8a6d8980106dec`.

X5 uses 1×Bayes-e. The eight-thread condition below applies to X3 measurements.

Published performance on RDK X5 and RDK X5 Module:

| Model       | Size    | Categories | Parameter | Floating point precision | Quantization accuracy | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ----------- | ------- | ---------- | --------- | ------------------------ | --------------------- | ------------------------------------ | ----------------------------------- | --------------- |
| RepViT_m1_1 | 224x224 | 1000       | 8.2       | 77.73                    | 77.50                 | 2.32                                 | 6.69                                | 590.42          |
| RepViT_m1_0 | 224x224 | 1000       | 6.8       | 76.75                    | 76.50                 | 1.97                                 | 5.71                                | 692.29          |
| RepViT_m0_9 | 224x224 | 1000       | 5.1       | 76.32                    | 75.75                 | 1.65                                 | 4.37                                | 902.69          |

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
| RepViT | 224x224 | 1000 | 5.1    | 75.25 | 75.75 | 28.34       | 41.22       | 96.47 |

Description:
1. X3 is in the best state: CPU is 4xA53@1.5G, full core Performance scheduling, BPU is 2xBernoulli@1G, a total of 5TOPS equivalent int8 computing power.
2. Single-thread latency measures one frame on one BPU core.
3. Four-thread FPS uses four concurrent task submissions, with BPU utilization at 100%.
4. The X3 eight-thread measurement uses eight concurrent task submissions to its two BPU cores.
5. Floating-point/fixed-point precision: Floating-point accuracy uses the Top-1 inference accuracy Level of onnx before the model is quantized, while quantized accuracy is the accuracy Level of the actual inference of the model after quantization.

<a id="boundaries"></a>
## Dataset-level evaluation

For dataset Top-1 accuracy, pass each validation image to the runtime entry through `--test-img`, compare the returned Top-1 class ID with that image’s ground-truth model index, and divide correct predictions by the number of labeled images evaluated. Keep the artifact, resize mode, Top-K, board image and scheduling settings fixed when comparing runs. For latency or FPS, time the inference stage on the matching board and record the thread count and operating mode alongside the result.
