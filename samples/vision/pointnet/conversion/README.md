English | [简体中文](README_cn.md)

# PointNet conversion notes

<a id="source-model"></a>
## Source model and architecture

The source delivery refers to [the S100 PointNet project](https://gitee.com/chenguanzhong/rdk_-s100_-point-net_-official).
It does not pin a training commit, framework version, checkpoint identifier or
weight checksum. The runtime's published HBM is not a reproducible conversion recipe.
PointNet applies shared MLP layers, symmetric max pooling and local/global feature
concatenation for per-point labels. The delivered task has four chair parts.

![Architecture](../test_data/readme_img/image-1.png)
![Segmentation network](../test_data/readme_img/image.png)

<a id="toolchain-targets"></a>
## Toolchain and target

| Target | march / OE version | Compile config |
| --- | --- | --- |
| s100 | not pinned in source recipe | absent |

Conversion belongs on an x86 Linux host with a compatible OpenExplore environment,
not inside the board inference wrapper. [OE resources](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)
and the [toolchain manual](https://toolchain.d-robotics.cc/) provide environment
references; they do not fill the sample-specific missing recipe.

<a id="export"></a>
## ONNX export

No checkpoint/export script or pinned export environment is included. The source
operator notes list Conv, BatchNorm and ReLU; preserve the graph below as a
historical illustration, not as proof that every operator of a new export is supported.
Expected runtime boundary is float32 `(1,3,N)` points and `(1,N,4)` part logits.
N is fixed by the actual compiled model, not by a user-selectable runtime option.

![Historical ONNX graph](../test_data/readme_img/char_static.png)

<a id="calibration"></a>
## Calibration

No calibration dataset, subset size, preparation script or configuration was
supplied. The source records int16 quantization and reports “trans > 0.9999” and
“pred > 0.98”; the metric definition and full measurement conditions are absent,
so these values are not a segmentation accuracy claim.

![Historical quantization record](../test_data/readme_img/pixpin_2025-07-07_20-44-37.jpg)

<a id="compile"></a>
## Compilation

No compile configuration/command exists for reproducing this HBM. Obtain the
published file through the [model preparation guide](../model/README.md).
Do not substitute an arbitrary generic compiler command and call it verified.

<a id="validation"></a>
## Validation boundary

```bash
# cwd: repository root; on S100 with the published HBM already prepared
python3 samples/vision/pointnet/runtime/python/main.py --target s100 --no-plot --output-dir outputs/pointnet-check
```

This is a functional smoke command, currently not-run for the unified entry.
For a newly compiled artifact, first establish its provenance, target and tensor
contract; passing shape checks alone cannot establish equivalence to the published
model. Compare point IDs against a pinned floating reference using the same input
order and normalization before making an accuracy claim.

<a id="artifacts"></a>
## Artifact

| File | Target | Default location |
| --- | --- | --- |
| pointnet.hbm | s100 | `samples/vision/pointnet/model/s100/pointnet.hbm` |

<a id="known-gaps"></a>
## Known gaps

Missing: pinned training source/checkpoint, export code/environment, calibration
set/config, compiler version/march/config, publisher checksum and reference
accuracy protocol. This directory retains useful architecture/operator records;
it does not offer end-to-end conversion. Code license:
[Apache-2.0](../../../../LICENSE); external model/source terms remain separate.
