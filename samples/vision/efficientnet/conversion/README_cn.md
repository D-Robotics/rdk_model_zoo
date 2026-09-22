# EfficientNet 转换

本目录合并了两套源配方。S 侧（rdk_s @380e1a2）是完整配方——导出脚本、
校准脚本与逐变体 YAML，且数值互相一致（迁移中已核对）；X5 侧
（rdk_x5 @ac11571）只交付参考 PTQ YAML，其缺口列在
[已知缺口](#known-gaps)，**不**构成可复现流程。本次迁移未执行任何转换
（未运行 OpenExplorer 环境）；见[验证](#validation)。

<a id="source-model"></a>
## 源模型

- S（lite0..lite4）：经 `timm` 导出的 EfficientNet-Lite 权重
  （`tf_efficientnet_lite0.in1k` .. `tf_efficientnet_lite4.in1k`），即源
  交付引用的 TensorFlow TPU EfficientNet-Lite 系列
  （<https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet>）。
- X5（b2/b3/b4）：EfficientNet B2/B3/B4。源交付只记录了参考性的 timm
  导出流程（create_model → torch.onnx.export → onnxsim.simplify），**未
  提供导出脚本，也未固定权重版本**；ONNX 来源因此未经核实。

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机上的目标平台 OpenExplorer Docker 内执行，
从不在板卡上运行。

- S100：march `nash-e`（交付 YAML 中的取值）。
- S600：同一配置将 march 改为 `nash-p`（改 YAML 或传工具链的 march
  覆盖参数），此为源交付的说明；量化配置其余不变。
- X5：march `bayes-e`。

- OE 资源入口（Docker + 开发包）：
  <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE 工具链手册：<https://toolchain.d-robotics.cc/>

<a id="export"></a>
## 导出（S 配方）

cwd：本 `conversion/` 目录。先安装导出依赖（在合适的 Python 3 环境中
`pip install timm onnx onnxsim`），再运行对应变体的导出脚本，例如：

```bash
# 输入：timm 权重 tf_efficientnet_lite0.in1k（由 timm 下载）
# 输出：./tf_efficientnet_lite0.onnx（opset 11，经 onnxsim 化简）
# 成功判据：脚本打印参数量并输出 "Simplified model is valid."
python3 get_efficientnet_lite0_onnx.py
```

| 变体 | 导出脚本 | ONNX 文件 | 前缀（== Manifest 文件名） | 输入 |
| --- | --- | --- | --- | --- |
| lite0 | `get_efficientnet_lite0_onnx.py` | `tf_efficientnet_lite0.onnx` | `efficientnet_lite0_224x224_nv12` | 224x224 |
| lite1 | `get_efficientnet_lite1_onnx.py` | `tf_efficientnet_lite1.onnx` | `efficientnet_lite1_240x240_nv12` | 240x240 |
| lite2 | `get_efficientnet_lite2_onnx.py` | `tf_efficientnet_lite2.onnx` | `efficientnet_lite2_260x260_nv12` | 260x260 |
| lite3 | `get_efficientnet_lite3_onnx.py` | `tf_efficientnet_lite3.onnx` | `efficientnet_lite3_300x300_nv12` | 300x300 |
| lite4 | `get_efficientnet_lite4_onnx.py` | `tf_efficientnet_lite4.onnx` | `efficientnet_lite4_380x380_nv12` | 380x380 |

`timm2onnx_local.py` 是源交付的替代辅助脚本：从本地下载的权重文件
（而非 timm hub）导出；需按目标变体修改其中的 `model_name`。

<a id="calibration"></a>
## 校准（S 配方）

cwd：本 `conversion/` 目录。一个脚本服务全部五个变体：

```bash
# 输入：ILSVRC2012_val_*.JPEG 图像（脚本使用排序后的前 100 张）
# 输出：./calibration_data_rgb/*.npy（float32），与 YAML 的 cal_data_dir 一致
# 成功判据：打印 "成功生成 ... 个校准数据文件"（列出 100 个文件）
python3 get_calibration_data.py
```

预处理链为 `ShortSideResize(224) → CenterCrop(224) → HWC2CHW →
Scale(255.0) → Mean([127,127,127]) → Scale(0.007843)`，已与每份 YAML 的
`mean_value: 127 127 127` / `scale_value: 0.007843 0.007843 0.007843`
交叉核对——脚本与 YAML 一致（与 ResNet152 源配方脚本/YAML 不一致的情况
不同；本 sample 不存在该差异）。

运行前需要知道两个源配方事实：

1. 脚本默认的 `src_image_dir` 指向 rdk_s 源码树内的 OpenExplorer 校准
   目录（`../../../open_explorer/samples/ai_toolchain/.../calibration_data/imagenet/`）。
   该相对路径在本仓库中**无法**解析——目录不存在。运行前请把
   `src_image_dir` 改为你自己的 ILSVRC2012 验证集 JPEG 目录。
2. resize/crop 对**所有**变体固定为 224（包括 240/260/300/380 的
   lite1..lite4）；源配方对全部变体复用同一套校准数据。此处按原样
   记录，未重新验证。

<a id="compile"></a>
## 编译（S 配方）

cwd：本 `conversion/` 目录，位于 OE Docker 内，且已完成导出与校准。按
源交付：

```bash
# S100 构建（输入：./tf_efficientnet_lite0.onnx + ./calibration_data_rgb）
# 输出：./model_output/efficientnet_lite0_224x224_nv12.hbm
hb_compile --config efficientnet_lite0_config.yaml
```

S600 需先把 YAML 中 `march` 从 `nash-e` 改为 `nash-p`（或传工具链的
march 覆盖参数）。输出前缀与 Manifest 文件名完全一致，产出的 `.hbm`
无需改名即可复制到 `model/s100/` 或 `model/s600/`。只有修改源模型或
转换配置时才需要重新构建——运行时 sample 下载的是已发布制品。

X5 YAML 用对应的 X5 OE 流程编译（`hb_mapper`/`hb_compile` + 配置
文件）；未固定的部分见[已知缺口](#known-gaps)第 5 条。

<a id="validation"></a>
## 验证

- `x86_inference.py`（S 配方）在 OE 环境内的 x86 主机上运行
  ONNX/HBIR/HBM 参考推理（导入 `horizon_tc_ui`）；用于把编译产物与
  ONNX 浮点模型对照。
- 板上功能检查走统一运行时：
  `python3 samples/vision/efficientnet/runtime/python/main.py --target s100 --asset-id s:efficientnet:s100/efficientnet_lite0_224x224_nv12.hbm ...`
  （见 [runtime/python/README_cn.md](../runtime/python/README_cn.md)）。
- **本次迁移未运行：**未执行导出、校准、编译或 x86 对照（未使用 OE
  环境）。上文 S 配方的一致性结论是脚本 ↔ YAML 数值与文件名/前缀同
  Manifest 吻合的静态交叉核对。

<a id="artifacts"></a>
## 保留材料

全部 16 个文件按源分支原字节保留（X5 @ac11571：三份 YAML；S
@380e1a2：五份 YAML、五个导出脚本、校准脚本、`timm2onnx_local.py`、
`x86_inference.py`）。其 SHA-256 由 `tests/test_conversion_layout.py`
钉住，今后对转换文件的任何改动都会被主机套件发现。原样保留的显著
内容：S YAML 使用 `working_dir: './model_output'`、
`calibration_type: 'max'`、`optimize_level: 'O2'`；X5 YAML 使用
`calibration_type: 'default'`、`compile_mode: 'latency'`、
`optimize_level: 'O3'`，且 B2/B4 携带 B3 没有的 `node_info` int16
配置。

<a id="known-gaps"></a>
## 已知缺口

X5 配方是参考配置，不是经过验证的可复现流程。按源交付原样保留：

1. **X5 无 ONNX 导出脚本。** `./efficientnet_b2.onnx` /
   `efficientnet_b3.onnx` / `efficientnet_b4.onnx` 没有生成脚本，也未
   固定权重或 timm 版本；记录的 timm 流程仅供参考。
2. **X5 无校准数据生成脚本。** YAML 期望 `./calibration_data_rgb_f32`
   （float32 RGB `.npy`），但源树中没有脚本产出它。等价数据必须遵循
   YAML 数值（mean `123.675 116.28 103.53`、scale `0.01712475
   0.017507 0.01742919`、224x224）；这是声明的要求，不是经验证的
   流水线。
3. **X5 输出前缀不含变体名。** 三份 YAML 的
   `output_model_file_prefix` 都是 `'EfficientNet_224x224_nv12'`，
   编译产物不带 B2/B3/B4 身份，在同一工作目录连续构建会互相覆盖；
   B3 的 `working_dir` 还是 `'model_output'`，与 B2/B4 的
   `'EfficientNet_224x224_nv12'` 不同。要复现 Manifest 制品，需把产出
   `.bin` 改名为 Manifest 名称（`EfficientNet_B2_224x224_nv12.bin`、
   ...）或先按变体改写前缀。
4. 三份 X5 YAML 均含 **`debug_mode: 'dump_calibration_data'`**（原样
   保留；其对已发布构建的影响未在此重新验证）。
5. **X5 无固定编译命令。** 源 README 指向通用 OE 流程
   （`hb_mapper checker` / `hb_mapper makertbin` / `hb_compile`）；
   产出已发布 `.bin` 的确切命令没有记录，X5 复现未经核实。
6. **本次迁移未执行转换**（见[验证](#validation)）；特别是 S 侧对
   240/260/300/380 模型使用 224 裁剪校准数据这一事实（校准节第 2 条）
   未重跑或数值确认。
