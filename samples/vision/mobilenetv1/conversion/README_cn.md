# MobileNetV1 模型转换

在 x86 Linux 主机的 RDK OpenExplore (OE) 环境中执行模型转换。
重建前准备模型图、校准数据和与目标匹配的 PTQ 配置。

<a id="source-model"></a>
## 源模型

MobileNetV1（[论文](https://arxiv.org/abs/1704.04861)，
[tensorflow/models](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)），
固定 NCHW 输入 `[1,3,224,224]`，ImageNet-1k 类别数。X5 使用所选
MobileNetV1 权重准备 ONNX 图；S 侧转换说明将源模型记为
MobileNet-Caffe，使用 S100 OE 工具链转换。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

使用与目标板卡匹配的 OE Docker/工具链版本，并记录镜像 tag、OE 版本与
主机日期。权威入口：
[RDK S 工具链总览](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)、
[D-Robotics 工具链下载](https://toolchain.d-robotics.cc/)。
目标：X5 用 `hb_mapper` 编译，march `bayes-e`；S100 用 `hb_compile`，
march `nash-e`；S600 为 `nash-p`。容器挂载仓库到 `/workspace` 并给足共享
内存（`--shm-size=15g`）。

<a id="export"></a>
## 导出

从所选上游 MobileNetV1 权重导出 ONNX 图，输入为 `[1,3,224,224]`，
输出为 1,000 类。随图记录框架、导出器与权重修订。

<a id="calibration"></a>
## 校准

按所选模型图的图像预处理准备校准集，并记录图像选择、归一化和
PTQ 配置。

<a id="compile"></a>
## 编译

为所选目标与模型图创建 OE 配置，使输入协议符合运行时契约
（X5 packed NV12，S100/S600 split Y/UV），并绑定上文准备的校准数据。

<a id="validation"></a>
## 验证

对再生成制品，按 OE 手册执行 `hb_perf` 与 `hrt_model_exec` 并保留完整
输出；随后在匹配板卡上用 运行时确认契约：X5 暴露一个 packed
NV12 输入与 F32 `[1,1000,1,1]` 输出；S100/S600 暴露 Y `[1,224,224,1]`、
UV `[1,112,112,2]` 与 F32 `[1,1000]` 输出；输出语义为softmax 之后的概率。
再生成制品需按 Validation 步骤重新验证。

<a id="artifacts"></a>
## 产物

推理时使用 [model/README_cn.md](../model/README_cn.md) 中的 Manifest 制品。

<a id="known-gaps"></a>
## 补充准备

推理请使用 [model/README_cn.md](../model/README_cn.md) 中按 Manifest 准备的制品。重新构建需提供 MobileNetV1 ONNX 图（224x224 RGB/NCHW 输入、1,000 类输出）、匹配的权重修订、OE PTQ 配置，以及符合模型预处理的校准数据。转换工作区未包含这些输入时，使用发布制品路线。
