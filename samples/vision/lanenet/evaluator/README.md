[English](README.md) | [简体中文](README_cn.md)

# LaneNet evaluation boundaries

The source evaluator directory contains only a placeholder. This migration provides reproducible host contract checks; it does not invent a dataset evaluator, instance-clustering algorithm or accuracy result.

<a id="dataset"></a>
## Dataset

The retained [lane.jpg](../test_data/lane.jpg) is a demonstration input, not a labeled evaluation set. No train/validation split, annotation conversion, dataset checksum or license record is supplied by the source evaluator. Four historical display PNGs are not ground truth. A future dataset evaluation must first define labels, split, preprocessing and lane-instance matching rules.

<a id="environment"></a>
## Environment

Host checks need Python, NumPy, OpenCV, PyYAML and a C++17 compiler. Native unit fixtures compile SDK-independent helpers and the actual resource owner against fake headers. They need no board, HBM, OE installation or real vendor library. Full native SDK/OpenCV compilation is a separate unperformed check. For actual inference environments, follow [Python](../runtime/python/README.md) or [C++](../runtime/cpp/README.md).

<a id="command"></a>
## Commands

Run from the repository root:

```bash
python3 -m unittest discover -s samples/vision/lanenet/tests
```

This executes host fixtures only. It does not download an asset or contact a board. The native tests compile and run temporary host executables using the local C++ compiler. To prepare raw results later on S100, use the runtime commands linked above with a new result directory for each implementation.

<a id="metrics"></a>
## What is measured

Host tests check source preprocessing arithmetic, metadata rejection, exact typed raw-output retention, independent result ownership, binary 0/1 constraints, display rounding, output collision rejection and conversion preparation. Native fixtures additionally exercise padded byte strides, overflow checks, role ambiguity, int64 serialization above 2^53, and resource cleanup across injected failures. These are implementation contracts, not lane-detection metrics.

A future board comparison must record exact model/image digests, board/runtime versions, full output metadata and raw arrays. Compare embeddings numerically with a justified declared tolerance, labels exactly, and visualizations separately. Python binds names while native binds shape/type; establish the actual role correspondence first. No tolerance, board equality or accuracy threshold is claimed from host fixtures.

<a id="outputs"></a>
## Outputs and evidence

Unit tests report their checks to the terminal. Runtime output formats are documented in the [Python results](../runtime/python/README.md#results) and [native results](../runtime/cpp/README.md#results-interpretation): preserve the NPZ name map or native role indices, not just screenshots. The source audit is recorded in [the migration report](../../../../docs/releases/unified-migration/2026-09-26-b8-lanenet-source-review.md).

<a id="reference-results"></a>
## Reference results

The source conversion documentation records an HRT run of 200 frames, 14.245 ms and 69.894 FPS. The associated board image, runtime/toolchain versions and artifact digest are not pinned; this is a historical source record, not a reproduced result or Python/C++ end-to-end latency. [Conversion notes](../conversion/README.md) retain its context and missing prerequisites.

At the current host implementation stage, 22 sample tests pass, including three native helper/resource/serialization fixtures and four native launcher tests. This count describes host coverage only. Real board inference, full native SDK build, OE compilation, dataset accuracy and performance remain **not-run**. Re-run the command above to obtain results for your checkout rather than relying on a static count.

<a id="boundaries"></a>
## Boundaries

No accuracy number, lane-instance metric, runtime speedup or cross-language board equivalence has been established. The source accuracy illustration path is missing and has not been replaced with fabricated evidence. Embedding display colors are not lane-instance assignments. Adding clustering, curve fitting or dataset scoring requires a defined algorithm and separate validation; it cannot be inferred from a passing host test suite.
