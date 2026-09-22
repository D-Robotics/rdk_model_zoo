# MobileNetV2 模型转换

模型转换在 x86 Linux 主机上的 RDK OpenExplore (OE) 环境中执行，不是板卡
操作。本目录保留源分支随附的转换材料，并如实记录缺口；不虚构能产出不同
制品的配置。

<a id="source-model"></a>
## 源模型

MobileNetV2（[论文](https://arxiv.org/abs/1801.04381)，
[timm/models/mobilenetv2](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv2.py)）。
两侧源分支都没有随附 ONNX 导出器；S 侧配置消费的 `mobilenetv2.onnx`
无法由源材料复现。

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

无法由本仓库复现（已知缺口）：两个源分支都没有为已发布 MobileNetV2 制品
提供 ONNX 导出器或权重出处。要重新生成 `.onnx` 需先针对上游模型编写导出；
在那之前，本目录只记录部署制品，不复现它们。
<a id="calibration"></a>
## 校准

S 侧配置指向 `../calibration_data_bgr`（float32）；校准集未随仓库
提供。再生成时必须先记录所用图像清单与预处理，其产物才能与已发布制品
比较。
<a id="compile"></a>
## 编译

本目录保留的参考配置：

| 配置 | 目标 | 命令（OE 容器内） |
| --- | --- | --- |
| `mobilenetv2_config.yaml` | s100 | `hb_compile --config mobilenetv2_config.yaml` |

S600 变体只需把 S 侧 YAML 中的 march 改为 `nash-p`。上表 march 取自 YAML
文件本身（X5 `bayes-e`，S100 `nash-e`）。本仓库未重跑编译；在对比目标、
输入 metadata、输出 shape/dtype 与数值结果之前，再生成制品不等价于已发布
制品。
<a id="validation"></a>
## 验证

对再生成制品，按 OE 手册执行 `hb_perf` 与 `hrt_model_exec` 并保留完整
输出；随后在匹配板卡上用 canonical 运行时确认契约：X5 暴露一个 packed
NV12 输入与 F32 `[1,1000,1,1]` 输出；S100/S600 暴露 Y `[1,224,224,1]`、
UV `[1,112,112,2]` 与 F32 `[1,1000]` 输出；输出语义为softmax 之后的概率。
本仓库对再生成制品的验证状态：**not-run**。

<a id="artifacts"></a>
## 保留材料

- `mobilenetv2_config.yaml`

<a id="known-gaps"></a>
## 已知缺口

- 两侧均无 ONNX 导出器，且无 X5 侧配置；仅保留 S 侧参考 YAML。
