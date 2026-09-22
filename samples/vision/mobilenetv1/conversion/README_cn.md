# MobileNetV1 模型转换

模型转换在 x86 Linux 主机上的 RDK OpenExplore (OE) 环境中执行，不是板卡
操作。本目录保留源分支随附的转换材料，并如实记录缺口；不虚构能产出不同
制品的配置。

<a id="source-model"></a>
## 源模型

MobileNetV1（[论文](https://arxiv.org/abs/1704.04861)，
[tensorflow/models](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)），
固定 NCHW 输入 `[1,3,224,224]`，ImageNet-1k 类别数。X5 源未提供导出
脚本（其 `conversion/` 只有 README）；S 侧转换说明将源模型记为
MobileNet-Caffe，使用 S100 OE 工具链转换。

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

无法由本仓库复现（已知缺口）：两个源分支都没有为已发布 MobileNetV1 制品
提供 ONNX 导出器或权重出处。要重新生成 `.onnx` 需先针对上游模型编写导出；
在那之前，本目录只记录部署制品，不复现它们。
<a id="calibration"></a>
## 校准

无法由本仓库复现（已知缺口）：两个分支都没有随附 MobileNetV1 的校准
集、预处理记录或 PTQ 配置。
<a id="compile"></a>
## 编译

两个源分支都没有随附 MobileNetV1 的 OE 配置（已知缺口）。已发布的
`.bin`/`.hbm` 制品在本仓库材料之外构建；要再生成，需要编写输入协议与
运行时契约一致的 OE 配置（X5 packed NV12，S100/S600 split Y/UV）并记录
校准数据。
<a id="validation"></a>
## 验证

对再生成制品，按 OE 手册执行 `hb_perf` 与 `hrt_model_exec` 并保留完整
输出；随后在匹配板卡上用 canonical 运行时确认契约：X5 暴露一个 packed
NV12 输入与 F32 `[1,1000,1,1]` 输出；S100/S600 暴露 Y `[1,224,224,1]`、
UV `[1,112,112,2]` 与 F32 `[1,1000]` 输出；输出语义为softmax 之后的概率。
本仓库对再生成制品的验证状态：**not-run**。

<a id="artifacts"></a>
## 保留材料

- (none — README-only record)

<a id="known-gaps"></a>
## 已知缺口

- 两侧分支都未提供导出器、OE 配置或校准集；转换只被记录，未被复现。
