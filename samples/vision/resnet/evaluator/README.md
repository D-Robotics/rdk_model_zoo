English | [简体中文](README_cn.md)

# ResNet evaluation (ResNet18/50/152)

Use the bundled image for a board classification. For dataset accuracy, prepare the matching labeled validation split and aggregate predictions using the metric definitions below. Reference tables list published measurements and their operating conditions.

<a id="dataset"></a>

## Dataset

Functional checks run on the bundled test images. For dataset-level
evaluation, prepare ImageNet validation data (50,000 images, ILSVRC2012
val) following [datasets/imagenet](../../../../datasets/imagenet/README.md);
the dataset is user-supplied (no download script ships here).

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="environment"></a>
## Environment

Run on the matching board with its `hbm_runtime` image, NumPy, OpenCV-Python, PyYAML, a prepared model artifact, and the label file. See the runtime README for installation. Record the board image and toolchain versions with dataset results.

<a id="command"></a>
## Command

Board classification — X5 (prerequisite:
`bash samples/vision/resnet/model/download.sh x5 resnet18`; output: exit 0
and the Top-K classes and scores):

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

Board classification — S100 (S600 uses `s600` in the asset reference and
artifact path):

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:resnet18:s100/resnet18_224x224_nv12.hbm \
  --model-path samples/vision/resnet/model/s100/resnet18_224x224_nv12.hbm \
  --test-img samples/vision/resnet/test_data/zebra_cls.jpg \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

ResNet50/152 (S100/S600) follow the same command with their
`s:resnet50|resnet152:<target>/...` references, matching artifact paths and
`--variant resnet50|resnet152`. The S-series C++ check runs
`bash samples/vision/resnet/runtime/cpp/run.sh`.

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| Top-K output | class IDs, scores and labels printed for the bundled image; repeated runs return the same list | same board, artifact bytes, image, resize type, Top-K |
| Top-1 accuracy | fraction of argmax-correct predictions | run on the prepared ImageNet val set; record the split with the result |
| latency / FPS | inference timing | measure on the target board; record batch/thread settings with the result (see the reference conditions below) |

<a id="outputs"></a>
## Outputs

The board command prints Top-K class IDs, scores, and labels to stdout. Use `--img-save-path` to save an annotated image. For comparisons, record the board, model, command, input image, resize mode, and raw scores with the results.

<a id="reference-results"></a>
## Reference results

Published X5 release record for ResNet18 (x5-v1.1.3):

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS | Threading conditions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | 224x224 | 1000 | 11.2 | 71.5% | 70.5% | 2.95 | 449+ | — (not stated by the source) |

ImageNet-val accuracy and latency are measured by preparing the dataset
above and recording the toolchain, batch and thread settings with the
result. The S release published no accuracy or latency figures for
ResNet18/50/152.

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
## Boundaries

Use your own evaluation loop with the prepared ImageNet validation data to measure dataset accuracy and latency. Record the board, artifact, toolchain, batch size, and thread settings with each result.