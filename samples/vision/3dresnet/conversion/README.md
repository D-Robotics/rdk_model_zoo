English | [简体中文](README_cn.md)

# R3D-18 Conversion Notes

<a id="source-model"></a>
## Source Model

The source notes describe a PyTorch `torchvision.models.video.r3d_18` action-classification model exported to ONNX for a 400-class Kinetics output. The network input is the prepared clip `(1,3,16,112,112)`; the runtime sample consumes the already normalized clip in `test_data/video0.npy`.

The official paper and reference implementation are linked from the [sample overview](../README.md#overview). No source checkpoint filename, checkpoint hash, export script, or source-weight acquisition record is present in this directory.

The original graph screenshot is retained:

![R3D-18 ONNX graph](../test_data/readme_img/r3d_18_orig.png)

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and Targets

The source record names the RDK S algorithm toolchain OpenExplorer 3.5.0 and the S100 target. It reports that `Conv3D` was supported while the original 3D `GlobalAveragePooling` path was not. The recorded workaround replaced that pooling path with an equivalent 2D `ReduceMean` before HBM compilation.

Only `s100/r3d_18.hbm` is published in the active manifest. No S100P, S600, or x5 conversion target is provided.

Model conversion runs on an x86 Linux host inside the RDK S100 OpenExplore environment, never on the board:

- OE Docker documentation: <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain download: <https://toolchain.d-robotics.cc/>

Download the OpenExplore CPU Docker image for RDK S100/S100P from the OE Docker documentation, then load the image file:

```bash
sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
sudo docker images
```

Start the container with the repository mounted and enough shared memory for compilation:

```bash
sudo docker run -it --rm --network host --shm-size=15g \
  -v "$(pwd)":/workspace --workdir /workspace \
  <docker-image-name> /bin/bash
```

<a id="export"></a>
## Export

The source README describes an ONNX export concept but provides no executable export script, Python environment lock, checkpoint path, or command with verifiable output. The export stage is therefore **not reproducible from repository contents** and is listed as a known gap.

<a id="calibration"></a>
## Calibration

No calibration dataset, sample count, quantization configuration, calibration command, or generated calibration artifact is present. The source conversion record reports most operator similarity values above 0.99 and a final quantization similarity around 0.99.

<a id="compile"></a>
## Compile

No compiler YAML, mapper, compile command, workspace, or checkpoint-to-HBM recipe is present. The only executable preparation path in this repository downloads the already published HBM artifact:

```bash
# cwd: repository root
bash samples/vision/3dresnet/model/download.sh s100
# expect: samples/vision/3dresnet/model/s100/r3d_18.hbm
```

This command prepares a published artifact; it does not perform conversion.

<a id="validation"></a>
## Post-conversion Validation



Board-side execution of the downloaded artifact uses the [Python runtime](../runtime/python/README.md) on an S100 board.

<a id="artifacts"></a>
## Artifacts

| Artifact | Target | Path after preparation | Availability |
| --- | --- | --- | --- |
| `r3d_18.hbm` | S100 | `model/s100/r3d_18.hbm` | published/downloadable; conversion recipe unavailable |

No ONNX, checkpoint, calibration, or compiler workspace artifact is supplied.

<a id="known-gaps"></a>
## Additional preparation

- No executable ONNX export script or pinned source checkpoint.
- No calibration dataset, sample count, quantization YAML, or calibration command.
- No OE compile YAML, mapper, or output-generation command.
- No reproducible conversion output hash; the active HBM manifest SHA is `null`.
- No S100P, S600, or x5 artifact.

The four source conversion screenshots remain available for traceability:

![Original pooling error](../test_data/readme_img/image-1.png)
![Original 3D pooling](../test_data/readme_img/image.png)
![Pooling replacement](../test_data/readme_img/image-2.png)
![Conversion result](../test_data/readme_img/image-3.png)
