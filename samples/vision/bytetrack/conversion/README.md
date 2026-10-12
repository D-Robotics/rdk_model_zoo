English | [简体中文](README_cn.md)

# ByteTrack conversion

<a id="source-model"></a>
## Source model

ByteTrack runs association on the CPU; only the YOLOv5 detector requires conversion. X5 uses 640x640 packed-NV12 BIN models, and S targets use 672x672 split-NV12 HBM models.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets

For X5, follow the [YOLOv5 conversion recipe](../../yolov5/conversion/README.md), including v2.0/v7.0 export, calibration data preparation, and NV12 compilation settings.

Convert S models on an x86 Linux host with RDK S OpenExplorer. Select the compilation target for your board; model paths are `yolov5x_672x672_nv12.hbm` under `s100/`, `s100p/`, or `s600/`. See the [OE overview](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview) and [toolchain manual](https://toolchain.d-robotics.cc/) for setup. Prepare the S export script and YAML configuration separately.

<a id="export"></a>
## Export

ByteTrack association runs on the CPU. Prepare the detector using the [YOLOv5 export procedure](../../yolov5/conversion/README.md#export): choose matching v2.0 or v7.0 weights with 640x640 input for X5, or a 672x672 YOLOv5x detector for S targets. Keep the three detection heads.

<a id="calibration"></a>
## Calibration

For X5, follow the [YOLOv5 calibration procedure](../../yolov5/conversion/README.md#calibration). For S targets, use the calibration configuration for the selected detector and toolchain. Include pedestrians, occlusions, and lighting conditions representative of the application.

<a id="compile"></a>
## Compile

For X5, follow the [YOLOv5 compilation procedure](../../yolov5/conversion/README.md#compile) with `yolov5_detect_bayese_640x640_nv12.yaml` to produce a packed-NV12 BIN. For S targets, compile the external ONNX, target-specific YAML, and calibration data in the matching RDK S OpenExplorer environment to produce a split-NV12 HBM. Input dimensions and detection-head layout must match the selected runtime variant.

<a id="validation"></a>
## Model interface

The X5 model accepts one packed-NV12 input; S models accept separate Y and UV inputs. Outputs are three YOLOv5 detection heads at different scales. Select the matching board, input dimensions, and model variant when running inference.

<a id="artifacts"></a>
## Artifacts

See [model preparation](../model/README.md) for X5 BIN and S-target HBM filenames and paths. ByteTrack CPU association runs from the bundled source.

<a id="known-gaps"></a>
## Additional preparation

Download X5, S100 and S600 detectors with the [preparation commands](../model/README.md#preparation). For S100P, prepare a matching nash-m detector locally. S conversion requires an external YOLOv5x 672×672 ONNX exporter, target-specific YAML and calibration data; a complete S conversion script and YAML are not included in this directory. Select the input video with `--input`.
