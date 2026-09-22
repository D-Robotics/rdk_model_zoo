# MobileNetV3 模型转换

模型转换在 x86 Linux 主机上的 RDK OpenExplore (OE) 环境中执行，不是板卡
操作。本目录保留源分支随附的转换材料，并如实记录缺口；不虚构能产出不同
制品的配置。

<a id="source-model"></a>
## 源模型

timm `mobilenetv3_large_100` 预训练权重（MobileNetV3-Large），由
`get_mobilenetv3_onnx.py` 固定；两个平台均为 NCHW 输入
`[1,3,224,224]`。

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
cwd 为 `samples/vision/mobilenetv3/conversion`：

```bash
# 输入：timm 预训练权重（未缓存时下载）
# 输出：下列 .onnx 文件 — 成功判据：onnx 化简检查通过
python3 get_mobilenetv3_onnx.py    # -> ./mobilenetv3_large_100.onnx
```

导出器使用 onnx-simplifier 并打印参数量。迁移期间本仓库未重跑该命令；
请将其视为源分支记录的配方，而非已验证结果。
<a id="calibration"></a>
## 校准

`get_calibration_data.py` 按源分支原样保留。其硬编码事实：从旧目录树
的源目录
（`../../../open_explorer/samples/ai_toolchain/horizon_model_convert_sample/01_common/calibration_data/imagenet/`，
该路径在本仓库不存在——请改为自备的 ImageNet 验证集目录）读取
`ILSVRC2012_val_*.JPEG`；变换链固定为 BGR（padded center crop 224、
resize、HWC→CHW、`RGB2BGRTransformer`、×255、mean
`103.94 116.78 123.68`、×0.017），输出到 `./calibration_data_bgr/`。
校准图像未随附；再生成时必须记录所用图像清单。

各 YAML 消费的内容 vs 脚本产出：

| 配置（目标） | `cal_data_dir` | YAML 声明的布局/mean | 原样脚本能产出？ |
| --- | --- | --- | --- |
| `mobilenetv3_s_config.yaml`（s100；s600 仅改 march） | `./calibration_data_bgr` | BGR，mean `103.53 116.28 123.675` | **能。** 改源目录后，输出目录与 BGR 链即匹配该配方。脚本 mean 常量（`103.94 116.78 123.68`）与 YAML 的 `103.53 116.28 123.675` 略有出入；两文件均为源分支原样——差异在此披露，不做静默修正。 |
| `MobileNetV3_config.yaml`（x5） | `./calibration_data_rgb_f32` | RGB，mean `123.675 116.28 103.53` | **不能——缺失前提。** 脚本没有 RGB 输出模式，且 X5 制品的 RGB 校准配方从未发布。把 `calibration_data_bgr` 改名为 `calibration_data_rgb_f32` 只会给 RGB 配置喂入 mean 顺序错误的 BGR 数据——改名不是修复。X5 校准步骤因此无法在本目录现状下复现。 |

本仓库未重跑 PTQ 校准。
<a id="compile"></a>
## 编译

本目录保留的参考配置：

| 配置 | 目标 | 配置引用的输入 | 命令（OE 容器内） |
| --- | --- | --- | --- |
| `MobileNetV3_config.yaml` | x5 | `./mobilenetv3_large_100.onnx`（与导出器输出一致）、`./calibration_data_rgb_f32`（**缺失**，见[校准](#calibration)） | `hb_mapper makertbin --config MobileNetV3_config.yaml` |
| `mobilenetv3_s_config.yaml` | s100（s600：march 改 `nash-p`） | `./mobilenetv3_large_100.onnx`（一致）、`./calibration_data_bgr`（改源目录后由脚本产出） | `hb_compile --config mobilenetv3_s_config.yaml` |

上表 march 取自 YAML 文件本身（X5 `bayes-e`，S100 `nash-e`）。本仓库未重跑
编译；在对比目标、输入 metadata、输出 shape/dtype 与数值结果之前，再生成
制品不等价于已发布制品。
重命名说明：S 源文件名为 `mobilenetv3_config.yaml`；在不区分大小写
的文件系统上与 X5 的 `MobileNetV3_config.yaml` 冲突，因此此处 S 副本
更名为 `mobilenetv3_s_config.yaml`——仅文件名变化，内容原样。

<a id="validation"></a>
## 验证

对再生成制品，按 OE 手册执行 `hb_perf` 与 `hrt_model_exec` 并保留完整
输出；随后在匹配板卡上用 canonical 运行时确认契约：X5 暴露一个 packed
NV12 输入与 F32 `[1,1000,1,1]` 输出；S100/S600 暴露 Y `[1,224,224,1]`、
UV `[1,112,112,2]` 与 F32 `[1,1000]` 输出；输出语义为原始 logits（softmax 由运行时任务施加）。
本仓库对再生成制品的验证状态：**not-run**。

<a id="artifacts"></a>
## 保留材料

- `get_mobilenetv3_onnx.py`
- `timm2onnx_local.py`
- `get_calibration_data.py`
- `MobileNetV3_config.yaml`
- `mobilenetv3_s_config.yaml`

<a id="known-gaps"></a>
## 已知缺口

- S 侧 YAML 因不区分大小写文件系统而更名
（`mobilenetv3_s_config.yaml`）；内容原样。
- **X5 校准无配方**：`MobileNetV3_config.yaml` 消费 RGB 的
`./calibration_data_rgb_f32`，保留的脚本产不出（校准器仅 BGR），源分支
也未发布过。X5 链路止步于该缺失前提。
- 校准脚本硬编码的源目录属于旧目录树；运行前只需改源目录一行，
其余不动。
- 校准脚本的 mean 常量与 S YAML 的 `mean_value` 略有出入（源分支原样的
不一致，上文已披露）。
- 端到端再生成（导出 → 校准 → 编译 → 板端验证）未在本仓库执行。
