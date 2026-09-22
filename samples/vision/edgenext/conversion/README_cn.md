# EdgeNeXt 转换

本目录是 rdk_x5 @ac11571 的逐字节迁入交付：四份参考 PTQ YAML
（`EdgeNeXt_{base,small,x_small,xx_small}_config.yaml`）。X5 源交付
**不带导出脚本和校准数据产出脚本**——这是参考配置集，不是可复现流程；
缺口见[已知缺口](#known-gaps)。本次迁移未执行任何转换（未运行
OpenExplorer 环境）。

与同族多数 sample 不同，EdgeNeXt 的配置是**正向锚定**的：每份 YAML 的
`output_model_file_prefix` 与其清单基名逐字一致，`onnx_model` 文件名也
携带自身变体——按配置编译直接产出已发布文件名，无需重命名。
`tests/test_conversion_layout.py` 钉住这一一致性。

<a id="source-model"></a>
## 源模型

EdgeNeXt base/small/x-small/xx-small（论文 [EdgeNeXt: Efficiently
Amalgamated CNN-Transformer Architecture for Mobile Vision
Applications](https://arxiv.org/abs/2206.10589)，参考实现
[mmaaz60/EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt)）。各 YAML 要求
`./edgenext_{base,small,x_small,xx_small}.onnx`，但源交付未记录导出配方、
未固定权重——ONNX 来源未核实。

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机的 RDK X5 OpenExplorer Docker 内执行
（march `bayes-e`），从不在板上运行。源 README 指向通用 OE 流程
（`hb_mapper makertbin`）；离线 Docker 镜像可从地瓜机器人开发者论坛获取。

<a id="export"></a>
## 导出

交付不含导出脚本。重新生成 ONNX 输入需自行复现上游 EdgeNeXt 导出；
产物须命名为 `edgenext_<variant>.onnx` 并放在本目录（或调整 YAML 的
`onnx_model`）。此步骤在本仓未验证。

<a id="calibration"></a>
## 校准

交付不含校准数据产出脚本。四份 YAML 均要求 `./calibration_data_rgb_f32`
（float32 RGB `.npy`）并使用 `calibration_type: 'max'`、
`max_percentile: 0.999`。等价数据必须遵循 YAML 数值（mean `123.675
116.28 103.53`，scale `0.01712475 0.017507 0.01742919`，224x224）；
这是声明的要求，不是已验证的管线。

<a id="compile"></a>
## 编译

在 OE 环境内、两个输入就绪后（以 base 为例；其余变体换用各自配置）：

```bash
# cwd：本转换目录
# 输入：./edgenext_base.onnx + ./calibration_data_rgb_f32
# 输出：working_dir 'EdgeNeXt_base_224x224_nv12'，产出
#       EdgeNeXt_base_224x224_nv12.bin——与清单名一致，无需重命名
hb_mapper makertbin --config EdgeNeXt_base_config.yaml
```

四份 YAML 均设 `compile_mode: 'latency'` / `optimize_level: 'O3'`，并通过
`node_info` 将交叉协方差注意力（xca）Softmax 节点以 int16 I/O 摆上 BPU
——每份 3 处（stages 1/2/3）；xx-small 另加 13 处（共 16 处）。均不带
`debug_mode` 与 `set_all_nodes_int16`。

<a id="validation"></a>
## 验证

交付不含 x86 参考脚本。功能检查即板上的统一运行时：
`python3 samples/vision/edgenext/runtime/python/main.py --target x5 --asset-id x5:edgenext:EdgeNeXt_base_224x224_nv12.bin ...`
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

1. **无 ONNX 导出脚本。** 四个 `edgenext_<variant>.onnx` 输入均无产出
   脚本、固定权重或已记录的导出配方。
2. **无校准数据产出脚本。** `./calibration_data_rgb_f32` 在源树中没有
   生成脚本。
3. **无固定编译命令。** 源 README 指向通用 OE 流程；产出各已发布
   `.bin` 的确切命令无记录，复现未经验证。
4. **本次迁移未执行转换。**

影响同族多个 sample 的"无变体输出前缀需重命名"缺口（convnext、
fasternet、fastvit 及 B2 的 efficientnet/efficientformer）在此**不适用**：
输出前缀携带自身变体名。
