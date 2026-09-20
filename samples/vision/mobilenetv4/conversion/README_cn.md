# MobileNetV4 模型转换

模型转换在 x86 Linux 主机上的 RDK OpenExplore (OE) 环境中执行，不是板卡
操作。本目录保留源分支随附的转换材料，并如实记录缺口；不虚构能产出不同
制品的配置。

<a id="source-model"></a>
## 源模型

timm `mobilenetv4_conv_small` 与 `mobilenetv4_conv_medium` 预训练
权重，由 `get_mobilenetv4_onnx.py` 固定。脚本导出 small 为
`[1,3,224,224]`、medium 为 `[1,3,256,256]`——X5 medium 的几何差异
见[已知缺口](#known-gaps)。

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

在 OE 容器内（或任一装有 `torch`、`timm`、`onnx`、`onnxsim` 的主机），
cwd 为 `samples/vision/mobilenetv4/conversion`：

```bash
# 输入：timm 预训练权重（未缓存时下载）
# 输出：下列 .onnx 文件 — 成功判据：onnx 化简检查通过
python3 get_mobilenetv4_onnx.py    # -> mobilenetv4_conv_small.onnx + mobilenetv4_conv_medium.onnx
```

导出器使用 onnx-simplifier 并打印参数量。迁移期间本仓库未重跑该命令；
请将其视为源分支记录的配方，而非已验证结果。
<a id="calibration"></a>
## 校准

`get_calibration_data.py` 是源分支的通用校准脚本：加载原始图像、执行
脚本内定义的预处理、写出供 YAML 消费的 `.npy` 校准样本（`cal_data_dir`：
per-config calibration directories (see each YAML)）。校准图像本身未随附；再生成时必须记录所用图像清单与
预处理。本仓库未重跑 PTQ 校准。
<a id="compile"></a>
## 编译

本目录保留的参考配置：

| 配置 | 目标 | 命令（OE 容器内） |
| --- | --- | --- |
| `MobileNetV4_small.yaml` | x5 | `hb_mapper makertbin --config MobileNetV4_small.yaml` |
| `MobileNetV4_medium.yaml` | x5 | `hb_mapper makertbin --config MobileNetV4_medium.yaml` |
| `mobilenetv4_small_config.yaml` | s100 | `hb_compile --config mobilenetv4_small_config.yaml` |
| `mobilenetv4_medium_config.yaml` | s100 | `hb_compile --config mobilenetv4_medium_config.yaml` |

S600 变体只需把 S 侧 YAML 中的 march 改为 `nash-p`。上表 march 取自 YAML
文件本身（X5 `bayes-e`，S100 `nash-e`）。本仓库未重跑编译；在对比目标、
输入 metadata、输出 shape/dtype 与数值结果之前，再生成制品不等价于已发布
制品。
几何说明：S 侧 medium 配置按 `mobilenetv4_medium_config.yaml` 记录的
256x256 输入编译；X5 medium 配置构建已发布的 224x224 制品。两种
几何都是真实存在的，运行时契约表按目标分别记录。

<a id="validation"></a>
## 验证

对再生成制品，按 OE 手册执行 `hb_perf` 与 `hrt_model_exec` 并保留完整
输出；随后在匹配板卡上用 canonical 运行时确认契约：X5 暴露一个 packed
NV12 输入与 F32 `[1,1000,1,1]` 输出；S100/S600 暴露 Y `[1,224,224,1]`、
UV `[1,112,112,2]` 与 F32 `[1,1000]` 输出；输出语义为原始 logits（softmax 由运行时任务施加）。
本仓库对再生成制品的验证状态：**not-run**。

<a id="artifacts"></a>
## 保留材料

- `get_mobilenetv4_onnx.py`
- `timm2onnx_local.py`
- `get_calibration_data.py`
- `MobileNetV4_small.yaml`
- `MobileNetV4_medium.yaml`
- `mobilenetv4_small_config.yaml`
- `mobilenetv4_medium_config.yaml`
- `x86_medium_inference.py`

<a id="known-gaps"></a>
## 已知缺口

- X5 侧导出脚本生成的 medium ONNX 为 256x256，而已发布 X5 medium 制品
及其配置为 224x224；源材料未记录两者如何调和。因此按字节可比地再生成
X5 medium 制品未被证明。
- `x86_medium_inference.py` 是审计 medium 变体时使用的主机 ONNX 参考，
不是部署路径。
- 端到端再生成未在本仓库执行。
