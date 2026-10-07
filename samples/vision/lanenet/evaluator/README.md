[English](README.md) | [简体中文](README_cn.md)

# LaneNet evaluation boundaries

This sample provides reproducible host contract checks for evaluation. Dataset evaluation, instance clustering and accuracy results require the labeled dataset and matching rules described below.

<a id="dataset"></a>
## Dataset

Use the bundled [lane.jpg](../test_data/lane.jpg) as the demonstration input and the four display PNGs as visualization examples. For dataset evaluation, prepare labeled images and record the train/validation split, annotation conversion, dataset checksum, license, preprocessing and lane-instance matching rules.

<a id="environment"></a>
## Environment

Host checks need Python, NumPy, OpenCV, PyYAML and a C++17 compiler. Native unit fixtures compile SDK-independent helpers and the actual resource owner against fake headers. They need no board, HBM, OE installation or real vendor library. Build the native SDK/OpenCV executable in the matching target environment. For actual inference environments, follow [Python](../runtime/python/README.md) or [C++](../runtime/cpp/README.md).

<a id="command"></a>
## Commands

Run from the repository root:

```bash
python3 -m unittest discover -s samples/vision/lanenet/tests
```

This executes host fixtures only. It does not download an asset or contact a board. The native tests compile and run temporary host executables using the local C++ compiler. To prepare raw results later on S100, use the runtime commands linked above with a new result directory for each implementation.

<a id="metrics"></a>
## What is measured

Host tests check preprocessing arithmetic, metadata rejection, exact typed raw-output retention, independent result ownership, binary 0/1 constraints, display rounding, output collision rejection and conversion preparation. Native fixtures additionally exercise padded byte strides, overflow checks, role ambiguity, int64 serialization above 2^53, and resource cleanup across injected failures. These are implementation contracts, not lane-detection metrics.

For a board comparison, record exact model/image digests, board/runtime versions, full output metadata and raw arrays. Compare embeddings numerically with a justified declared tolerance, labels exactly, and visualizations separately. Python binds names while native binds shape/type; establish the actual role correspondence first.

<a id="outputs"></a>
## Outputs and evidence

Unit tests report checks to the terminal. Keep the runtime NPZ name map or native output-role indices with saved arrays; formats are documented in [Python results](../runtime/python/README.md#results) and [C++ results](../runtime/cpp/README.md#results-interpretation).

<a id="reference-results"></a>
## Reference results

The published HRT reference measurement uses 200 frames: 14.245 ms model latency and 69.894 FPS. Its board image, runtime/toolchain versions and artifact digest are unspecified. Measure Python/C++ end-to-end latency in the target environment and record those conditions with the result. See the [conversion guide](../conversion/README.md) for model preparation.

The sample suite contains 22 tests, including three native helper/resource/serialization fixtures and four native launcher tests, covering host behavior. Run the command above for development checks; perform inference, native SDK builds, OE compilation, dataset evaluation and performance measurements in their matching environments.

<a id="boundaries"></a>
## Boundaries

Dataset accuracy and lane-instance metrics are measured on a labeled dataset: define the lane-instance matching rules, run a defined clustering and curve-fitting step over the embeddings, then score the results against the dataset labels. Embedding display colors are a visual rendering feature; lane-instance identity comes from the chosen clustering/fitting algorithm. Runtime speedups and cross-language board comparisons are recorded from their own runs in the target environments.
