English | [简体中文](README_cn.md)

# UNetMobileNet conversion boundaries

<a id="source-model"></a>
## Source model

The S source ships precompiled HBM and an architectural description, but no training framework version, checkpoint, exact training repository or export code. U-Net/MobileNet papers explain the family; they do not reconstruct this deployed weight file.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Published target files are S100 and S600 only. Platform profiles identify nash-e and nash-p respectively, but no per-model OE version, YAML or compiler configuration is supplied. S100P has no published asset; no Nash-M recipe is implied.

<a id="export"></a>
## Export

No executable export recipe exists. Required before adding one: licensed trained checkpoint, exact architecture/revision, export dependencies, tensor names and validated static graph. Do not treat a downloaded HBM as an ONNX export input.

<a id="calibration"></a>
## Calibration

No calibration dataset, normalization policy, preprocessing recipe or quantization configuration is supplied. Runtime BGR→NV12 input handling is separate from the training normalization; confirm the normalization behind your artifacts. The two supplied pictures are not a representative calibration set.

<a id="compile"></a>
## Compile

No model compilation command is provided because required source inputs/configuration are missing. Native runtime CMake compilation is a different operation; see [C++ build](../runtime/cpp/README.md#build). Use the [published model preparation](../model/README.md#preparation) for current inference.

<a id="validation"></a>
## Validation

Future converted models must demonstrate target identity, Y [1,1024,2048,1]/UV [1,512,1024,2] uint8, NHWC [1,H,W,19] int32 or F32 scores and correct quantization descriptors. Match preprocessing and compare decoded masks against a trusted reference on real inputs; exact host fixtures alone are insufficient. ONNX/BPU numerical comparison runs with the validation steps on the board.

<a id="artifacts"></a>
## Artifacts

Current deliverables are the two externally downloaded HBM files, with unknown publisher SHA-256, and runtime outputs; no ONNX, checkpoint, compiler logs or calibration data is bundled. Record those provenance materials when a real conversion workflow is introduced.

<a id="known-gaps"></a>
## Additional preparation

Missing: exact checkpoint/architecture, export script/dependencies, calibration and normalization, OE version/target configs, compiled-model correspondence and real validation. These gaps are recorded explicitly rather than filled with generic toolchain commands.
