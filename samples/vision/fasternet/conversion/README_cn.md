# FasterNet 转换

本目录是 rdk_x5 @ac11571 的逐字节迁入交付：四份参考 PTQ YAML
（`FasterNet_{S,T0,T1,T2}_config.yaml`）。X5 源交付**不带导出脚本和校准
数据产出脚本**——这是参考配置集，不是可复现流程；缺口见
[已知缺口](#known-gaps)。本次迁移未执行任何转换（未运行 OpenExplorer
环境）。

两处源交付特异形态由 `tests/test_conversion_layout.py` 钉住而非"修复"：
四份配置共用**无变体**输出前缀 `FasterNet_224x224_nv12`；`working_dir`
互不一致（S 用 `model_output`，T0 加 `_mix` 后缀，T1/T2 用裸前缀）。

<a id="source-model"></a>
## 源模型

FasterNet S/T0/T1/T2（论文 [Run, Don't Walk: Chasing Higher FLOPS for
Faster Neural Networks](https://arxiv.org/abs/2303.03667)，按源交付引用
——源未记录参考实现链接）。各 YAML 要求
`./fasternet_{s,t0,t1,t2}.onnx`，但源交付未记录导出配方、未固定权重
——ONNX 来源未核实。

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机的 RDK X5 OpenExplorer Docker 内执行
（march `bayes-e`），从不在板上运行。源 README 指向通用 OE 流程
（`hb_mapper makertbin`）；离线 Docker 镜像可从地瓜机器人开发者论坛获取。

<a id="export"></a>
## 导出

交付不含导出脚本。重新生成 ONNX 输入需自行复现上游 FasterNet 导出；
产物须命名为 `fasternet_<variant>.onnx`（小写，与 YAML 一致）并放在
本目录（或调整 YAML 的 `onnx_model`）。此步骤在本仓未验证。

<a id="calibration"></a>
## 校准

交付不含校准数据产出脚本。四份 YAML 均要求 `./calibration_data_rgb_f32`
（float32 RGB `.npy`）并使用 `calibration_type: 'default'`。等价数据必须
遵循 YAML 数值（mean `123.675 116.28 103.53`，scale `0.01712475
0.017507 0.01742919`，224x224）；这是声明的要求，不是已验证的管线。

<a id="compile"></a>
## 编译

在 OE 环境内、两个输入就绪后（以 S 为例；其余变体换用各自配置）：

```bash
# cwd：本转换目录
# 输入：./fasternet_s.onnx + ./calibration_data_rgb_f32
# 输出：working_dir 'model_output'（S）/ 'FasterNet_224x224_nv12_mix'（T0）
#       / 'FasterNet_224x224_nv12'（T1/T2），产出
#       FasterNet_224x224_nv12.bin——需重命名，见缺口
hb_mapper makertbin --config FasterNet_S_config.yaml
```

四份 YAML 均设 `compile_mode: 'latency'` / `optimize_level: 'O3'`。仅
T0 配置通过 `node_info` 将节点以 int16 I/O 摆上 BPU（2 处 partial-conv
相关摆放）；S/T1/T2 无摆放。均不带 `debug_mode` 与
`set_all_nodes_int16`。

<a id="validation"></a>
## 验证

交付不含 x86 参考脚本。功能检查即板上的统一运行时：
`python3 samples/vision/fasternet/runtime/python/main.py --target x5 --asset-id x5:fasternet:FasterNet_S_224x224_nv12.bin ...`
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

1. **无 ONNX 导出脚本。** 四个 `fasternet_<variant>.onnx` 输入均无产出
   脚本、固定权重或已记录的导出配方。
2. **无校准数据产出脚本。** `./calibration_data_rgb_f32` 在源树中没有
   生成脚本。
3. **无变体输出前缀。** 四份 YAML 均产出
   `output_model_file_prefix: 'FasterNet_224x224_nv12'`，编译产物为
   `FasterNet_224x224_nv12.bin`，而非任何清单名
   （`FasterNet_{S,T0,T1,T2}_224x224_nv12.bin`）。要复现已发布制品，
   需先重命名产物或修改前缀。（ONNX 输入名携带小写变体；仅输出侧
   无变体。）
4. **working_dir 不一致。** S 编译进 `model_output`、T0 进
   `FasterNet_224x224_nv12_mix`、T1/T2 进 `FasterNet_224x224_nv12`——
   同一交付内三种约定，逐字保留。
5. **无固定编译命令。** 源 README 指向通用 OE 流程；产出各已发布
   `.bin` 的确切命令无记录，复现未经验证。
6. **本次迁移未执行转换。**
