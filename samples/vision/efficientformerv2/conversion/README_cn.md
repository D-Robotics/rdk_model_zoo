# EfficientFormerV2 转换

本目录是 rdk_x5 @ac11571 的原样交付：三份参考 PTQ YAML
（`EfficientFormerv2_s0_config.yaml`、`EfficientFormerv2_s1_config.yaml`、
`EfficientFormerv2_s2_config.yaml`）。X5 源交付**不带导出脚本，也不带校准
数据生产脚本**，因此这是参考配置而非可复现流程；缺口见
[已知缺口](#known-gaps)。本次迁移未执行任何转换（未运行 OpenExplorer
环境）。

<a id="source-model"></a>
## 源模型

EfficientFormerV2-S0/S1/S2（论文 [EfficientFormerV2: Rethinking Vision
Transformers for MobileNet Size and
Speed](https://arxiv.org/abs/2212.08059)）。三份 YAML 期望
`./efficientformerv2_s0.onnx`、`./efficientformerv2_s1.onnx`、
`./efficientformerv2_s2.onnx`，但源交付没有记录导出配方，也未固定权重
——ONNX 出处未经验证。

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机上的 RDK X5 OpenExplorer Docker（march
`bayes-e`）中执行，绝不在板卡上运行。源 README 指向通用 OE 流程
（`hb_mapper makertbin`）；离线 Docker 镜像可从 D-Robotics 开发者论坛
获取。

<a id="export"></a>
## 导出

本交付没有导出脚本。要重新生成 ONNX 输入，需自行复现上游
EfficientFormerV2 的导出；产物必须命名为 `efficientformerv2_s0.onnx`、
`efficientformerv2_s1.onnx`、`efficientformerv2_s2.onnx` 并放在本目录
（或修改 YAML 的 `onnx_model`）。本步骤未经此处验证。

<a id="calibration"></a>
## 校准

本交付没有校准数据生产脚本。三份 YAML 都期望
`./calibration_data_rgb_f32`（float32 RGB `.npy`），使用
`calibration_type: 'max'` 且逐变体 `max_percentile`：S0 与 S1 为
`0.999`，S2 为 `0.9995`。等价数据必须遵循 YAML 数值（mean
`123.675 116.28 103.53`，scale `0.01712475 0.017507 0.01742919`，
224x224）；该等价性是声明的要求，不是已验证的流水线。

<a id="compile"></a>
## 编译

在 OE 环境中，两个输入就绪后编译对应变体：

```bash
# cwd：本 conversion 目录
# 输入：./efficientformerv2_s0.onnx + ./calibration_data_rgb_f32
# 输出：working_dir 'EfficientFormerv2_s0_int16_model_output'，
#       产出的 .bin 按 output_model_file_prefix 命名（见下文）
hb_mapper makertbin --config EfficientFormerv2_s0_config.yaml
```

与 efficientnet/efficientformer 的 X5 交付不同，这里每份 YAML 都携带
变体身份：`output_model_file_prefix` 为
`EfficientFormerv2_s{0,1,2}_224x224_nv12`，产出的 `.bin` 名称直接复现
Manifest 基名（`EfficientFormerv2_s0_224x224_nv12.bin`……），无需改名，
且每个变体编译进各自的 `working_dir`，无冲突。三份 YAML 均设置
`compile_mode: 'latency'` / `optimize_level: 'O3'`，并携带 `node_info`
Softmax int16 摆放（S0/S1 为 5 个节点，S2 为 10 个）。S0 额外设置
`debug_mode: "dump_calibration_data"` 与
`optimization: "set_all_nodes_int16"`，S1/S2 没有——该不对称为源交付
原样保留，按原样记录。

<a id="validation"></a>
## 转换后验证

本交付没有 x86 参考脚本。功能检查是板上的统一运行时：
`python3 samples/vision/efficientformerv2/runtime/python/main.py --target x5 --asset-id x5:efficientformerv2:EfficientFormerv2_s0_224x224_nv12.bin ...`
（见 [runtime/python/README_cn.md](../runtime/python/README_cn.md)）。
**本次迁移未执行：**未运行任何导出、校准或编译；此处的一致性结论是对
YAML 内容及前缀/文件名与 Manifest 吻合关系的静态交叉核对。

<a id="artifacts"></a>
## 保留材料

三份 YAML 文件自 rdk_x5 @ac11571 逐字节保留；其 SHA-256 由
`tests/test_conversion_layout.py` 固定，任何后续修改都会被主机套件
捕获。

<a id="known-gaps"></a>
## 已知缺口

按源交付原样保留：

1. **无 ONNX 导出器。**三个 `efficientformerv2_s*.onnx` 输入都没有生产
   脚本、固定权重或成文导出配方。
2. **无校准数据生产脚本。**源码树中没有 `./calibration_data_rgb_f32`
   的生成脚本。
3. **S0 独有的 debug/optimization 不对称。**只有 S0 YAML 设置
   `debug_mode: "dump_calibration_data"` 与
   `optimization: "set_all_nodes_int16"`；源没有解释该不对称的原因。
   S0 的 `working_dir` 拼写（`EfficientFormerv2_s0_int16_model_output`）
   也与兄弟变体（`EfficientFormerv2_s{1,2}_224x224_nv12`）不同——输出
   前缀仍带变体身份，故此处仅是表面差异。
4. **无固定的编译命令。**源 README 指向通用 OE 流程；产出已发布
   `.bin` 的确切命令没有记录，复现未经验证。
5. **本次迁移未执行任何转换。**
