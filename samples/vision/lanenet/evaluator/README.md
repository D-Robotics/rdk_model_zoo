[English](README.md) | [简体中文](README_cn.md)

# LaneNet evaluation boundaries

Use this guide to compare LaneNet outputs and record model performance. Dataset metrics require labeled images and declared lane-instance matching rules.

<a id="dataset"></a>
## Dataset

Use the bundled [lane.jpg](../test_data/lane.jpg) as the demonstration input and the four display PNGs as visualization examples. For dataset evaluation, prepare labeled images and record the train/validation split, annotation conversion, dataset checksum, license, preprocessing and lane-instance matching rules.

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="environment"></a>
## Environment

Use the matching S100 HBM, board SDK, Python, NumPy, OpenCV and PyYAML. Build the native executable using the [C++ guide](../runtime/cpp/README.md) for C++ comparisons.

<a id="command"></a>
## Commands

After preparing the model on S100, run from the repository root:

```bash
bash samples/vision/lanenet/model/download.sh --target s100
python3 -m samples.vision.lanenet.runtime.python.main --target s100 --output outputs/lanenet-eval
```

Use a new output directory. Successful execution exits 0 and writes raw tensors and a report.

<a id="metrics"></a>
## Metrics

For the same model and image, compare embedding tensors with a declared numerical tolerance, compare binary labels exactly, and compare visualizations separately. Dataset evaluation requires defined clustering, curve fitting and lane-instance matching rules before scoring against annotations.

<a id="outputs"></a>
## Outputs and evidence

Keep the runtime NPZ name map or native output-role indices with saved arrays; formats are documented in [Python results](../runtime/python/README.md#results) and [C++ results](../runtime/cpp/README.md#results-interpretation).

<a id="reference-results"></a>
## Reference results

The published HRT reference measurement uses 200 frames: 14.245 ms model latency and 69.894 FPS. Its board image, runtime/toolchain versions and artifact digest are unspecified. Measure Python/C++ end-to-end latency in the target environment and record those conditions with the result. See the [conversion guide](../conversion/README.md) for model preparation.

<a id="boundaries"></a>
## Scope

Dataset accuracy and lane-instance metrics are measured on a labeled dataset: define the lane-instance matching rules, run a defined clustering and curve-fitting step over the embeddings, then score the results against the dataset labels. Embedding display colors are a visual rendering feature; lane-instance identity comes from the chosen clustering/fitting algorithm. Runtime speedups and cross-language board comparisons are recorded from their own runs in the target environments.
