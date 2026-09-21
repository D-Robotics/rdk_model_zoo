# FastViT 转换

本目录是 rdk_x5 @ac11571 的逐字节迁入交付：四份参考 PTQ YAML
（`FastViT_{S12,SA12,T12,T8}_config.yaml`）。X5 源交付**不带导出脚本和
校准数据产出脚本**——这是参考配置集，不是可复现流程；缺口见
[已知缺口](#known-gaps)。本次迁移未执行任何转换（未运行 OpenExplorer
环境）。

两处源交付特异形态由 `tests/test_conversion_layout.py` 钉住而非"修复"：
所有 `onnx_model` 指向本 sample 目录**之外的公共模型库路径**；四份配置
共用**无变体**输出前缀 `FastViT_224x224_nv12`（复现已发布基名需重命名）。

<a id="source-model"></a>
## 源模型

FastViT S12/SA12/T12/T8（论文 [FastViT: A Fast Hybrid Vision Transformer
using Structural
Reparameterization](https://arxiv.org/abs/2303.14189)，按源交付引用——
源未记录参考实现链接）。各 YAML 从共享 `01_common` 模型库消费 ONNX
（见缺口 1），源未记录导出配方、未固定权重——ONNX 来源未核实。

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机的 RDK X5 OpenExplorer Docker 内执行
（march `bayes-e`），从不在板上运行。源 README 指向通用 OE 流程
（`hb_mapper makertbin`）；离线 Docker 镜像可从地瓜机器人开发者论坛获取。

<a id="export"></a>
## 导出

交付不含导出脚本。按交付原样，各配置要求 ONNX 输入位于
`../../../01_common/model_zoo/mapper/classification/FastViT/fastvit_<variant>.onnx`
——本仓库不携带该目录。重新生成输入需自行复现上游 FastViT 导出，并
恢复该布局或调整 `onnx_model`。此步骤在本仓未验证。

<a id="calibration"></a>
## 校准

交付不含校准数据产出脚本。四份 YAML 均要求 `./calibration_data_rgb_f32`
（float32 RGB `.npy`）并使用 `calibration_type: 'default'`。等价数据必须
遵循 YAML 数值（mean `123.675 116.28 103.53`，scale `0.01712475
0.017507 0.01742919`，224x224）；这是声明的要求，不是已验证的管线。

<a id="compile"></a>
## 编译

在 OE 环境内、两个输入就绪后（以 S12 为例；其余变体换用各自配置）：

```bash
# cwd：本转换目录
# 输入：外部 01_common ONNX（见缺口 1）+ ./calibration_data_rgb_f32
# 输出：working_dir 'FastViT_224x224_nv12_mix'，产出
#       FastViT_224x224_nv12.bin——需重命名，见缺口
hb_mapper makertbin --config FastViT_S12_config.yaml
```

四份 YAML 均设 `compile_mode: 'latency'` / `optimize_level: 'O3'`，并通过
`node_info` 将重参数化注意力/MLP 节点以 int16 I/O 摆上 BPU
（S12/SA12/T12/T8 各 5/6/4/10 处）。均不带 `debug_mode` 与
`set_all_nodes_int16`。

<a id="validation"></a>
## 验证

交付不含 x86 参考脚本。功能检查即板上的统一运行时：
`python3 samples/vision/fastvit/runtime/python/main.py --target x5 --asset-id x5:fastvit:FastViT_S12_224x224_nv12.bin ...`
（见 [runtime/python/README_cn.md](../runtime/python/README_cn.md)）。
**本次迁移未运行**：未执行导出、校准或编译；本文的一致性结论是 YAML
内容与清单文件名/前缀的静态交叉核对。

<a id="artifacts"></a>
## 保留材料

四份 YAML 自 rdk_x5 @ac11571 逐字节保留；SHA-256 由
`tests/test_conversion_layout.py` 钉住，后续任何改动都会被主机套件捕获。

<a id="known-gaps"></a>
## 已知缺口

按源交付原样保留：

1. **ONNX 输入在外部。** 所有 `onnx_model` 指向本 sample 目录之外的
   `../../../01_common/model_zoo/mapper/classification/FastViT/...`；本
   仓库不携带该目录，任何输入都无产出脚本与固定权重。
2. **无校准数据产出脚本。** `./calibration_data_rgb_f32` 在源树中没有
   生成脚本。
3. **无变体输出前缀。** 四份 YAML 均产出
   `output_model_file_prefix: 'FastViT_224x224_nv12'`，编译产物为
   `FastViT_224x224_nv12.bin`，而非任何清单名
   （`FastViT_{S12,SA12,T12,T8}_224x224_nv12.bin`）。要复现已发布制品，
   需先重命名产物或修改前缀。
4. **无固定编译命令。** 源 README 指向通用 OE 流程；产出各已发布
   `.bin` 的确切命令无记录，复现未经验证。
5. **本次迁移未执行转换。**
