[English](README.md) | 简体中文

# ByteTrack 转换

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

S100 对应 `s100/yolov5x_672x672_nv12.hbm` Nash-e，S100P 对应自己的 `s100p/...`，S600 对应自己的 `s600/...`。转换需在 x86 Linux 主机使用 RDK S OpenExplorer；源资源为 [OE overview](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview) 和 [toolchain manual](https://toolchain.d-robotics.cc/)。sample 没有 S YAML 或 exporter。

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

使用已发布模型时，按 [模型准备步骤](../model/README_cn.md#preparation) 下载即可。自行转换 S 模型时需准备相应 ONNX、板卡 YAML 和校准数据。输入视频使用 `--input` 指定。
