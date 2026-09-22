# EfficientFormer 转换

本目录是 rdk_x5 @ac11571 的原样交付：两份参考 PTQ YAML
（`EfficientFormer_l1_config.yaml`、`EfficientFormer_l3_config.yaml`）。
X5 源**未提供导出脚本和校准数据生成脚本**，因此这是参考配置，不是可
复现流程；缺口列在[已知缺口](#known-gaps)。本次迁移未执行任何转换
（未运行 OpenExplorer 环境）。

<a id="source-model"></a>
## 源模型

EfficientFormer-L1 与 EfficientFormer-L3（论文 [EfficientFormer:
ImageNet Transformers at MobileNet
Speed](https://arxiv.org/abs/2206.00171)）。YAML 期望
`./efficientformer_l1.onnx` / `./efficientformer_l3.onnx`，但源交付没有
记录导出配方，也没有固定权重——ONNX 来源未经核实。

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机上的 RDK X5 OpenExplorer Docker 内执行
（march `bayes-e`），从不在板卡上运行。源 README 指向通用 OE 流程
（`hb_mapper checker` / `hb_mapper makertbin` / `hb_compile`）；离线
Docker 镜像可从 D-Robotics 开发者论坛获取。

<a id="export"></a>
## 导出

本交付没有导出脚本。要重新生成 ONNX 输入，需自行复现上游
EfficientFormer 导出；产物必须命名为 `efficientformer_l1.onnx` /
`efficientformer_l3.onnx` 并放在本目录（或修改 YAML 的 `onnx_model`）。
该步骤在此未经验证。

<a id="calibration"></a>
## 校准

本交付没有校准数据生成脚本。YAML 期望 `./calibration_data_rgb_f32`
（float32 RGB `.npy`）。等价数据必须遵循 YAML 数值（mean
`123.675 116.28 103.53`、scale `0.01712475 0.017507 0.01742919`、
224x224）；这是声明的要求，不是经验证的流水线。

<a id="compile"></a>
## 编译

在 OE 环境内、两个输入就绪后，编译对应变体：

```bash
# 输入：./efficientformer_l1.onnx + ./calibration_data_rgb_f32
# 输出前缀：EfficientFormer_224x224_nv12（不含变体名——见缺口）
hb_mapper makertbin --config EfficientFormer_l1_config.yaml
```

两份 YAML 均使用 `calibration_type: 'default'` 加
`optimization: "set_all_nodes_int16"`、`compile_mode: 'latency'` /
`optimize_level: 'O3'`；L3 另设 `jobs: 64`，且比 L1 携带更多
`node_info` Softmax int16 配置。

<a id="validation"></a>
## 验证

本交付没有 x86 参考脚本。功能检查走板上的统一运行时：
`python3 samples/vision/efficientformer/runtime/python/main.py --target x5 --asset-id x5:efficientformer:EfficientFormer_l1_224x224_nv12.bin ...`
（见 [runtime/python/README_cn.md](../runtime/python/README_cn.md)）。
**本次迁移未运行：**未执行导出、校准或编译；此处的一致性结论是 YAML
内容与文件名/前缀同 Manifest 吻合的静态交叉核对。

<a id="artifacts"></a>
## 保留材料

两份 YAML 按源 rdk_x5 @ac11571 原字节保留；其 SHA-256 由
`tests/test_conversion_layout.py` 钉住，今后的任何改动都会被主机套件
发现。

<a id="known-gaps"></a>
## 已知缺口

按源交付原样保留：

1. **无 ONNX 导出脚本。** `efficientformer_l1.onnx` 与
   `efficientformer_l3.onnx` 都没有生成脚本、固定权重或成文导出配方。
2. **无校准数据生成脚本。** `./calibration_data_rgb_f32` 在源树中没有
   生成脚本。
3. **输出前缀不含变体名。** 两份 YAML 的
   `output_model_file_prefix` 都是 `'EfficientFormer_224x224_nv12'`（且
   共用 `working_dir: 'EfficientFormer_224x224_nv12_int16'`），编译产物
   不带 L1/L3 身份，在同一工作目录连续构建会互相覆盖。要复现 Manifest
   制品，需把产出 `.bin` 改名为 Manifest 名称
   （`EfficientFormer_l1_224x224_nv12.bin`、...）或先按变体改写前缀。
4. **无固定编译命令。** 源 README 指向通用 OE 流程；产出已发布
   `.bin` 的确切命令没有记录，复现未经核实。
5. **本次迁移未执行转换。**
