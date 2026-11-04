[English](README.md) | 简体中文

# ByteTrack 转换

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
## 源模型

ByteTrack 的跟踪关联在 CPU 上运行，只需转换 YOLOv5 检测模型。X5 使用 640x640 packed-NV12 BIN，S 系列使用 672x672 split-NV12 HBM。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

X5 使用 [YOLOv5 转换配方](../../yolov5/conversion/README_cn.md)，其中包含 v2.0/v7.0 导出、校准数据准备和 NV12 编译配置。

S 系列模型需在 x86 Linux 主机上的 RDK S OpenExplorer 环境中转换。按板卡选择编译目标，模型路径分别为 `s100/`、`s100p/` 和 `s600/` 下的 `yolov5x_672x672_nv12.hbm`。环境配置见 [OE 概述](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)和[工具链手册](https://toolchain.d-robotics.cc/)。S 系列的导出脚本及 YAML 配置需另行准备。

<a id="export"></a>
## 导出

ByteTrack 的关联算法在 CPU 上运行。检测模型按 [YOLOv5 导出步骤](../../yolov5/conversion/README_cn.md#export) 准备：X5 选择匹配的 v2.0 或 v7.0 权重，输入尺寸为 640x640；S 系列使用 672x672 YOLOv5x。输出保留三个检测头。

<a id="calibration"></a>
## 校准

X5 按 [YOLOv5 校准步骤](../../yolov5/conversion/README_cn.md#calibration) 准备代表性图片及校准数据。S 系列使用对应工具链和检测模型的校准配置。校准集应包含目标应用中的行人、遮挡和光照变化。

<a id="compile"></a>
## 编译

X5 使用 [YOLOv5 编译步骤](../../yolov5/conversion/README_cn.md#compile) 及 `yolov5_detect_bayese_640x640_nv12.yaml` 生成 packed-NV12 BIN。S 系列在对应 RDK S OpenExplorer 环境中，将外部 ONNX、目标板卡 YAML 与校准集编译为 split-NV12 HBM。模型尺寸和检测头布局须与所选 Runtime 变体一致。

<a id="validation"></a>
## 模型接口

产物输入协议：X5 为单个 packed-NV12 输入，S 系列为分离的 Y/UV 输入。输出为三个尺度的 YOLOv5 检测头。板卡、输入尺寸及模型变体必须与运行参数匹配。

<a id="artifacts"></a>
## 产物

X5 BIN 和 S 系列 HBM 的文件名与路径见 [模型说明](../model/README_cn.md)。ByteTrack 的 CPU 关联算法随源码运行。

<a id="known-gaps"></a>
## 补充准备

X5、S100、S600 检测模型按[准备命令](../model/README_cn.md#preparation)下载。S100P 需准备匹配的本地 nash-m 检测模型。S 系列转换需要外部 YOLOv5x 672×672 ONNX 导出器、目标板卡 YAML 和校准数据；本目录未提供完整的 S 系列转换脚本及 YAML。输入视频使用 `--input` 指定。
