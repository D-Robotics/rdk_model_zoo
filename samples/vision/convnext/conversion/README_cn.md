# ConvNeXt 转换

本目录是 rdk_x5 @ac11571 的逐字节迁入交付：三份参考 PTQ YAML
（`ConvNeXt_atto.yaml`、`ConvNeXt_femto.yaml`、`ConvNeXt_nano.yaml`）。
仅 **atto** 有已发布清单资产；femto/nano 是无发布制品的配方。X5 源交付
**不带导出脚本和校准数据产出脚本**，且三份 YAML 的 ONNX 引用彼此错位
——这是参考配置集，不是可复现流程；缺口见[已知缺口](#known-gaps)。本次
迁移未执行任何转换（未运行 OpenExplorer 环境）。

<a id="source-model"></a>
## 源模型

ConvNeXt（论文 [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)，
参考实现 [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)）。
atto/femto/nano 尺寸来自官方 ConvNeXt 尺寸阶梯。各 YAML 的
`onnx_model` 与自身变体名**不一致**（见缺口 1）；每个配置的 ONNX 来源
均未核实。

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机的 RDK X5 OpenExplorer Docker 内执行
（march `bayes-e`），从不在板上运行。源 README 指向通用 OE 流程
（`hb_mapper makertbin`）；离线 Docker 镜像可从地瓜机器人开发者论坛获取。

<a id="export"></a>
## 导出

交付不含导出脚本。重新生成 ONNX 输入需自行复现上游 ConvNeXt 导出。
注意三份 YAML 指向**互为错位**的文件——按交付原样：atto 配置消费
`./convnext_femto.onnx`，femto 配置指向
`../../../01_common/model_zoo/mapper/classification/ConvNeXt/convnext_atto.onnx`
（本 sample 目录之外的路径），nano 配置指向 `./convnext_pico.onnx`
（交付中并无该尺寸的其它文件）。已发布 atto 制品实际由哪个文件产出
无记录；此步骤在本仓未验证。

<a id="calibration"></a>
## 校准

交付不含校准数据产出脚本。三份 YAML 均要求 `./calibration_data_rgb_f32`
（float32 RGB `.npy`）并使用 `calibration_type: 'default'`。等价数据必须
遵循 YAML 数值（mean `123.675 116.28 103.53`，scale `0.01712475
0.017507 0.01742919`，224x224）；这是声明的要求，不是已验证的管线。

<a id="compile"></a>
## 编译

在 OE 环境内、两个输入就绪后（以 atto 为例）：

```bash
# cwd：本转换目录
# 输入：该 YAML 的 onnx_model 目标（见缺口 1——atto 配置按交付原样指向
#       ./convnext_femto.onnx）+ ./calibration_data_rgb_f32
# 输出：working_dir 'ConvNeXt-deploy_224x224_nv12'，产出
#       ConvNeXt-deploy_224x224_nv12.bin——需重命名，见缺口
hb_mapper makertbin --config ConvNeXt_atto.yaml
```

三份 YAML 均设 `compile_mode: 'latency'` / `optimize_level: 'O3'`，并通过
`node_info` 将深度卷积/归一化节点以 int16 I/O 摆上 BPU（atto 8 处、
femto 5 处、nano 8 处；无 Softmax 摆放——ConvNeXt 没有注意力 softmax）。
均不带 `debug_mode` 与 `set_all_nodes_int16`。

<a id="validation"></a>
## 验证

交付不含 x86 参考脚本。功能检查即板上的统一运行时：
`python3 samples/vision/convnext/runtime/python/main.py --target x5 --asset-id x5:convnext:ConvNeXt_atto_224x224_nv12.bin ...`
（见 [runtime/python/README_cn.md](../runtime/python/README_cn.md)）。
**本次迁移未运行**：未执行导出、校准或编译；本文的一致性结论是 YAML
内容与清单文件名/前缀的静态交叉核对。

<a id="artifacts"></a>
## 保留材料

三份 YAML 自 rdk_x5 @ac11571 逐字节保留；SHA-256 由
`tests/test_conversion_layout.py` 钉住，后续任何改动都会被主机套件
捕获。

<a id="known-gaps"></a>
## 已知缺口

按源交付原样保留：

1. **ONNX 引用互为错位。** atto→`./convnext_femto.onnx`、
   femto→`../../../01_common/model_zoo/mapper/classification/ConvNeXt/convnext_atto.onnx`
   （sample 目录之外）、nano→`./convnext_pico.onnx`。源 README 的目录
   树还列出了**并未随交付**的 `ConvNeXt_pico.yaml`。此处不作任何"修复"；
   没有任何文件具备产出脚本、固定权重或已记录的导出配方。
2. **无校准数据产出脚本。** `./calibration_data_rgb_f32` 在源树中没有
   生成脚本。
3. **无变体输出前缀。** 三份 YAML 均产出
   `output_model_file_prefix: 'ConvNeXt-deploy_224x224_nv12'`，编译产物
   为 `ConvNeXt-deploy_224x224_nv12.bin`，而非清单名
   `ConvNeXt_atto_224x224_nv12.bin`。要复现已发布制品，需先重命名产物
   或修改前缀。
4. **femto/nano 无已发布资产。** 清单仅有 atto 一行；两份额外配方无法
   对照已发布制品验证，仅作为源材料保留。
5. **无固定编译命令。** 源 README 指向通用 OE 流程；产出已发布 `.bin`
   的确切命令无记录，复现未经验证。
6. **本次迁移未执行转换。**
