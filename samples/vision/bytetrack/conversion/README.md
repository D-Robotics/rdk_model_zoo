English | [简体中文](README_cn.md)

# ByteTrack conversion

## Rebuilt YOLOv5x v7 S profiles

The 2026-11-04 profile `yolov5x-v7-672-nv12-int16-cpuptq-o1-20261104` uses the official YOLOv5 v7 source commit `915bbf294bb74c859f0b41f1c23bc395014ea679` and trained `yolov5x.pt` SHA-256 `9f27a794fa0308e2606f90565164571d6f0a0ba18a3ce2e5e5d323b71c157859` (GPL-3.0). Each accepted target is listed in `converted_targets_20261104.json`; only these exact manifest IDs select the new raw-F32 ABI. The historical `x-672` defaults retain their old contracts.

```bash
# x86 Linux/dev-server only; source tree, weights and a pinned real 200-image manifest already present.
python3 -m samples.vision.bytetrack.conversion.prepare calibrate --images /path/to/coco-person-200 --output /path/to/new/calibration
python3 -m samples.vision.bytetrack.conversion.prepare export --source /path/to/yolov5-v7 --checkpoint /path/to/yolov5x.pt --target s100 --calibration /path/to/new/calibration --output /path/to/new/export
# In the cached S SDK container; use the generated config's absolute paths.
hb_compile -c /path/to/new/export/config.yaml
```

S100 / S100P / S600 require independent `nash-e` / `nash-m` / `nash-p` builds. Exact observed tools: hb_compile3.5.3, HMCT2.6.5, HBDK4.7.5; cached S SDK3.7.0 CPU image. INT16 max percentile0.9999, O1, one core, two jobs. Source input: RGB float32 `[1,3,672,672]`, stretch `INTER_NEAREST`, divide by255 once. Calibration stores raw0..255 RGB; compiler performs the normalization. Source eval raw heads are bit-exact before export, then all three NHWC80-class heads pass source/ORT checks on three real images and seed42 random input at atol1e-4/rtol1e-3. CPU SDK full-output gates passed separately. Actual HBM has split U8 Y/UV inputs and raw F32 heads at84/42/21; no second dequantization or legacy letterboxing.

The effective original recipe, trained source pair, calibration manifest, CPU gates, compiler receipt, actual HBM interface and complete original NodeInfo/OE reports are frozen per target. Board smoke, accuracy, Runtime1/2 and C++1/2 remain `not-run`. The exporter here is the subsequent portable entry; original Workbench producer hashes remain in the acceptance evidence. The sections below preserve historical artifact instructions.


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
