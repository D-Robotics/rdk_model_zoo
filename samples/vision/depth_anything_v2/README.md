[English](README.md) | [简体中文](README_cn.md)

# Depth Anything V2

<a id="overview"></a>
## Overview

Estimate dense relative depth from a single image using the published S100 HBM.
This sample separates preprocessing, inference and postprocessing from SDK
ownership, image IO and visualization. People and Agents use the same commands.
The result contains original-size float depth and an INFERNO display image;
values are relative, not calibrated meters.

The source describes V2's use of synthetic labeled training images, a larger
teacher and pseudo-labeled real images to improve fine detail and robustness.
Its framework figure and upstream references are retained as background, not a
new validation of this compiled artifact:

![Source framework](test_data/readme_img/image-2.png)

Source-listed references: [project](https://depth-anything.github.io/),
[paper](https://arxiv.org/abs/2406.19675),
[upstream repository](https://github.com/DepthAnything/Depth-Anything-V2).

<a id="support-matrix"></a>
## Support and validation

| Target | Published artifact | Implementation | Current verification |
| --- | --- | --- | --- |
| S100 | `s100/depth_any.hbm`; publisher digest unknown | Python | host fixtures only; board not-run |
| S100P | no separate manifest asset | explicitly refused | source prose mentions it; compatibility not established |
| S600 / X5 | no manifest asset | explicitly refused | no inferred substitute |

There is no source C++ implementation. Internal int16 quantization does not imply
an int16 public tensor: the source IO contract is RGB float32 `[1,3,518,686]` and
float32 depth `[1,518,686]`; metadata is checked at load. Actual artifact metadata
has not been observed in this host migration. Source claims and historical
records do not establish current board acceptance.

<a id="prerequisites"></a>
## Prerequisites

Use an S100 Linux image with compatible vendor `hbm_runtime`, Python, NumPy,
OpenCV and PyYAML. Host inspection needs no SDK. Runtime does not install
packages or download a model; prepare these explicitly. The source used Torch
only for resizing: this implementation uses OpenCV linear instead. Bit identity
with Torch is not claimed; see [runtime contracts](runtime/python/README.md).

<a id="quickstart"></a>
## Quick start

From repository root, inspect without a model or board:

```bash
python -m samples.vision.depth_anything_v2.runtime.python.main --list-models
python -m samples.vision.depth_anything_v2.runtime.python.main --target s100 --dry-run
```

Prepare explicitly, then run on the actual S100:

```bash
bash samples/vision/depth_anything_v2/model/download.sh --target s100
bash samples/vision/depth_anything_v2/runtime/python/run.sh --target s100 \
  --output outputs/depth-anything-s100
```

The wrapper resolves relative user paths from repository root. The output
directory must not exist. Default input is the bundled `furseal.jpg`. Optional
`--img-save-path result.jpg` adds a source-style color image; it must also be new.
`auto` selects the sole S100 asset, but execution still verifies the local board.
It cannot make S100P compatible by assigning an S100 filename.

<a id="expected-results"></a>
## Expected results and deliberate corrections

A successful run writes `raw_depth.npy`, `depth_native.npy`, `depth_gray.png`,
`depth_color.png` and `report.json`. The report binds local model/input hashes,
selection, runtime metadata and preprocessing policy. It measures no latency.
Unknown publisher hash/runtime version remain unknown.

![Historical source result](test_data/readme_img/depth_color.png)

This image is preserved source evidence, not output from this migration. Color
normalization is per image; similar colors do not establish equal depth or
accuracy. A constant map now gives zero grayscale instead of division by zero.

Actual source preprocessing is **pixelwise RGB z-score**, not the ImageNet
constants described in its docstring. Default resize remains nearest-neighbor
stretch. Optional letterbox retains source linear resize/fill127, but now crops
padding before restoration. Float task output replaces the old uint8 display API;
visualization is separate. These changes and compatibility limits are documented
in the [source audit](../../../docs/releases/unified-migration/2026-09-26-b8-depth-anything-source-review.md).

<a id="directory"></a>
## Directory

| Directory | Responsibility |
| --- | --- |
| [model](model/README.md) | Exact asset, explicit download, paths and unknown hash |
| [runtime/python](runtime/python/README.md) | Stages, SDK runner, CLI, rendering and provenance |
| [conversion](conversion/README.md) | Source ONNX/quantization facts and missing recipe prerequisites |
| [evaluator](evaluator/README.md) | Historical performance, interpretation and unverified dataset scope |
| test_data | Original furseal image and source explanatory/result figures |
| tests | Host fixtures; do not substitute for hardware evidence |

<a id="entry-points"></a>
## Entry points

Start with the commands above, then the runtime guide's API and stage table.
Conversion/evaluation pages preserve source detail while marking unavailable
inputs. Original [S source](../../../platforms/s/samples/vision/depth_anything_v2/README.md)
remains available; its implicit installation/download script and API are historical.
No board test, dataset score, conversion run or SDK compatibility acceptance is
claimed by this canonical entry.

<a id="license"></a>
## License

Sample code uses the repository [Apache 2.0 license](../../../LICENSE).
Upstream weights, training data and compiled artifacts retain their own applicable
terms; this page does not grant additional rights to them.
