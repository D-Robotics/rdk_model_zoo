English | [简体中文](README_cn.md)

# PointNet chair part segmentation

<a id="overview"></a>
## Overview

PointNet segments point clouds using shared MLPs and symmetric max pooling. This sample identifies four parts of a chair: back, seat, leg and arm, while keeping the input point order.

References: [PointNet paper](https://arxiv.org/abs/1612.00593), [official implementation](https://github.com/charlesq34/pointnet), [the S100 PointNet project](https://gitee.com/chenguanzhong/rdk_-s100_-point-net_-official).

<a id="directory"></a>
## Directory structure

```text
pointnet/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python and native inference implementations
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

 Board inference requires the S100 board SDK and the
published HBM; images/benchmarks below are source records.

<a id="support-matrix"></a>
## Support and validation

| Target | Variant | Python | C++ |
| --- | --- | --- | --- |
| s100 | chair, four parts | supported | not-supported |
| x5 / s100p / s600 | none published | not-supported | not-supported |


<a id="prerequisites"></a>
## Prerequisites

Use an RDK S100 with its matching `hbm_runtime` installation; do not install a
similarly named PyPI package as a replacement. Python 3.10+ is required for this
entry; NumPy and PyYAML are required, matplotlib is needed for plots. The source
does not pin a minimum firmware/SDK version or a measured memory/storage budget;
choose the firmware/SDK for your deployment and size memory/storage from a real
run. Keep space for the HBM and output files. Host help/list/dry-run do not
need the SDK or model file.

<a id="quickstart"></a>
## Quick start

Run from the repository root on S100. The download step is explicit and never
runs inside `predict`. Input `test_data/chair.pts` is already in the repository.
```bash
# cwd: repository root
python3 -m pip install numpy PyYAML matplotlib
bash samples/vision/pointnet/model/download.sh --target s100
python3 samples/vision/pointnet/runtime/python/main.py --target s100
```

Exit 0 and `outputs/pointnet/result.json` indicate the command completed. Inspect
`result.png` beside the original `result_orig.png`; the plot axes are X/Z/Y as
in the source visualization. The runtime checks the input point count against
the compiled model, without silent sampling or padding.

<a id="expected-results"></a>
## Expected results

`labels.npy` contains int32 `(N,)` part IDs, in input row order; `result.json`
contains point counts, normalization, asset identity and observed metadata.
Counts sum to N but need not contain all four parts for every input. No accuracy
threshold is inferred from a visually plausible segmentation.

![Reference chair result](test_data/readme_img/chair_res.png)

<a id="entry-points"></a>
## Entry points

[Model preparation](model/README.md) · [Python parameters and API](runtime/python/README.md)
· [Conversion](conversion/README.md) · [Evaluation](evaluator/README.md).
There is no C++ implementation for this sample.

<a id="license"></a>
## License

Code follows the repository [Apache-2.0 license](../../../LICENSE). The linked
training projects and model weights retain their own terms; the publisher
manifest does not supply a separate weight-license statement. Do not infer it
from the sample code license.
