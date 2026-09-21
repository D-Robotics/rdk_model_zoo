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

`get_calibration_data.py` 按源分支原样保留。其硬编码事实：与 V3 校准器
相同，从旧目录树源目录
（`../../../open_explorer/samples/ai_toolchain/horizon_model_convert_sample/01_common/calibration_data/imagenet/`，
本仓库不存在——请改为自备的 ImageNet 验证集目录）读取
`ILSVRC2012_val_*.JPEG`；变换链为 BGR（padded center crop、resize、
HWC→CHW、`RGB2BGRTransformer`、×255、mean `103.94 116.78 123.68`、
×0.017），图像尺寸由脚本内两行注释开关选择。校准图像未随附；再生成时
必须记录所用图像清单。

各 YAML 消费的内容 vs 脚本产出：

| 配置（目标） | `cal_data_dir` | YAML 声明的布局/尺寸 | 原样脚本如何产出 |
| --- | --- | --- | --- |
| `mobilenetv4_small_config.yaml`（s100；s600 仅改 march） | `./calibration_data_bgr_224` | BGR，224 | 默认即匹配：`output_calib_dir = './calibration_data_bgr_224/'` 与 `data_transformer(224)`——只需改源目录。 |
| `mobilenetv4_medium_config.yaml`（s100；s600 仅改 march） | `./calibration_data_bgr_256` | BGR，**256** | 按脚本自身注释把两行开关切到 `output_calib_dir = './calibration_data_bgr_256/'` 与 `active_transformers = data_transformer(256)`（默认激活的是 224 一对），并改源目录。两行必须一起切换——256 目录配 224 数据（或相反）都是错的。 |
| `MobileNetV4_small.yaml`（x5） | `./calibration_data_rgb_f32` | RGB，224 | **缺失前提**——脚本没有 RGB 输出模式；把 BGR 目录改名不是 RGB 配方。 |
| `MobileNetV4_medium.yaml`（x5） | `./calibration_data_rgb_f32` | RGB，224 | **缺失前提**——同上 RGB 缺口。 |

脚本 mean 常量（`103.94 116.78 123.68`）与 S 侧 YAML 的
`103.53 116.28 123.675` 略有出入；两文件均为源分支原样——在此披露，
不做静默修正。本仓库未重跑 PTQ 校准。
<a id="compile"></a>
## 编译

本目录保留的参考配置：

| 配置 | 目标 | 配置引用的输入 | 命令（OE 容器内） |
| --- | --- | --- | --- |
| `MobileNetV4_small.yaml` | x5 | `./mobilenetv4_conv_small.onnx`（与导出器输出一致）、`./calibration_data_rgb_f32`（**缺失**，见[校准](#calibration)） | `hb_mapper makertbin --config MobileNetV4_small.yaml` |
| `MobileNetV4_medium.yaml` | x5 | `./mobilenetv4_conv_medium_deploy.onnx`（**无保留脚本能产出该文件**）、`./calibration_data_rgb_f32`（**缺失**） | `hb_mapper makertbin --config MobileNetV4_medium.yaml` |
| `mobilenetv4_small_config.yaml` | s100（s600：march `nash-p`） | `./mobilenetv4_conv_small.onnx`（一致）、`./calibration_data_bgr_224`（脚本产出） | `hb_compile --config mobilenetv4_small_config.yaml` |
| `mobilenetv4_medium_config.yaml` | s100（s600：march `nash-p`） | `./mobilenetv4_conv_medium.onnx`（与导出器的 256 导出一致）、`./calibration_data_bgr_256`（两行开关切换后由脚本产出） | `hb_compile --config mobilenetv4_medium_config.yaml` |

**X5 medium 配置叠加三个缺失前提**：它读取的 `mobilenetv4_conv_medium_deploy.onnx`
没有任何保留脚本能产出（导出器写出的是 256x256 的
`mobilenetv4_conv_medium.onnx`，见[源模型](#source-model)），已发布 X5
medium 制品为 224x224，且其校准目录要求 RGB。把导出文件改名成 deploy
名字既不调和几何也不调和颜色顺序；X5 medium 链路无法在本目录现状下复现，
不对其声称无条件可执行的配方。

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
输出；随后在匹配板卡上用 canonical 运行时确认契约。输入 shape 按
目标×变体给出——与运行时契约表及已发布制品文件名一致：

| 目标 / 变体 | metadata 暴露的输入 | 输出 |
| --- | --- | --- |
| x5，small 与 medium | 单个 packed NV12 输入，224x224（`MobileNetV4_conv_{small,medium}_224x224_nv12.bin`） | F32 `[1,1000,1,1]` |
| s100/s600，small | Y `[1,224,224,1]`、UV `[1,112,112,2]`（`mobilenetv4_small_224x224_nv12.hbm`） | F32 `[1,1000]` |
| s100/s600，medium | **Y `[1,256,256,1]`、UV `[1,128,128,2]`**（`mobilenetv4_medium_256x256_nv12.hbm`） | F32 `[1,1000]` |

若用上表 224 的 shape 去验收正确的 S medium 制品属于验收错误——S medium
输入是 256x256。输出语义为原始 logits（softmax 由运行时任务施加）。
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
- X5 medium 配置还期望一个 `mobilenetv4_conv_medium_deploy.onnx` 输入，
无保留脚本能产出——见[编译](#compile)。
- **X5 校准（small 与 medium）无配方**：两个 X5 YAML 均消费 RGB 的
`./calibration_data_rgb_f32`，保留的脚本产不出（校准器仅 BGR），源分支
也未发布过。
- 校准脚本硬编码的源目录属于旧目录树；运行需改源目录，S medium 配方
还需把两行 256 开关注释一起切换。
- 校准脚本的 mean 常量与 S YAML 的 `mean_value` 略有出入（源分支原样的
不一致，上文已披露）。
- `x86_medium_inference.py` 是审计 medium 变体时使用的主机 ONNX 参考，
不是部署路径。
- 端到端再生成未在本仓库执行。
