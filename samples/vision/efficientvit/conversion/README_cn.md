# EfficientViT 转换

本目录是 rdk_x5 @ac11571 的原样交付：单份参考 PTQ YAML
（`EfficientViT_MSRA_m5_config.yaml`）。X5 源交付**不带导出脚本，也不带
校准数据生产脚本**，因此这是参考配置而非可复现流程；缺口见
[已知缺口](#known-gaps)。本次迁移未执行任何转换（未运行 OpenExplorer
环境）。

<a id="source-model"></a>
## 源模型

EfficientViT-MSRA m5（论文 [EfficientViT: Memory Efficient Vision
Transformer with Cascaded Group
Attention](https://arxiv.org/abs/2305.07027)，参考实现
[microsoft/Cream/EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT)）。
YAML 期望 `./efficientvit_m5.onnx`，但源交付没有记录导出配方，也未固定
权重——ONNX 出处未经验证。

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机上的 RDK X5 OpenExplorer Docker（march
`bayes-e`）中执行，绝不在板卡上运行。源 README 指向通用 OE 流程
（`hb_mapper makertbin`）；离线 Docker 镜像可从 D-Robotics 开发者论坛
获取。

<a id="export"></a>
## 导出

本交付没有导出脚本。要重新生成 ONNX 输入，需自行复现上游 MSRA
EfficientViT 的导出；产物必须命名为 `efficientvit_m5.onnx` 并放在本目录
（或修改 YAML 的 `onnx_model`）。本步骤未经此处验证。

<a id="calibration"></a>
## 校准

本交付没有校准数据生产脚本。YAML 期望 `./calibration_data_rgb_f32`
（float32 RGB `.npy`），使用 `calibration_type: 'max'` 且
`max_percentile: 0.99999`——X5 分类家族中最保守的分位值（兄弟样例为
0.999–0.9995）。等价数据必须遵循 YAML 数值（mean `123.675 116.28
103.53`，scale `0.01712475 0.017507 0.01742919`，224x224）；该等价性是
声明的要求，不是已验证的流水线。

<a id="compile"></a>
## 编译

在 OE 环境中，两个输入就绪后：

```bash
# cwd：本 conversion 目录
# 输入：./efficientvit_m5.onnx + ./calibration_data_rgb_f32
# 输出：working_dir 'EfficientViT_msra_224x224_nv12'，产出
#       EfficientViT_msra_224x224_nv12.bin——需改名，见缺口
hb_mapper makertbin --config EfficientViT_MSRA_m5_config.yaml
```

YAML 设置 `compile_mode: 'latency'` / `optimize_level: 'O3'`，并通过
`node_info` 将 28 个注意力 `Softmax` 节点以 int16 I/O 摆上 BPU（级联组
注意力结构）。与 EfficientFormerV2 交付不同，本配置不带 `debug_mode`，
也没有 `set_all_nodes_int16` 优化。

<a id="validation"></a>
## 转换后验证

本交付没有 x86 参考脚本。功能检查是板上的统一运行时：
`python3 samples/vision/efficientvit/runtime/python/main.py --target x5 --asset-id x5:efficientvit:EfficientViT_m5_224x224_nv12.bin ...`
（见 [runtime/python/README_cn.md](../runtime/python/README_cn.md)）。
**本次迁移未执行：**未运行任何导出、校准或编译；此处的一致性结论是对
YAML 内容及文件名/前缀与 Manifest 吻合关系的静态交叉核对。

<a id="artifacts"></a>
## 保留材料

YAML 文件自 rdk_x5 @ac11571 逐字节保留；其 SHA-256 由
`tests/test_conversion_layout.py` 固定，任何后续修改都会被主机套件
捕获。

<a id="known-gaps"></a>
## 已知缺口

按源交付原样保留：

1. **无 ONNX 导出器。**`efficientvit_m5.onnx` 没有生产脚本、固定权重或
   成文导出配方。
2. **无校准数据生产脚本。**源码树中没有 `./calibration_data_rgb_f32`
   的生成脚本。
3. **无变体的输出前缀。**YAML 产出
   `output_model_file_prefix: 'EfficientViT_msra_224x224_nv12'`，编译
   文件是 `EfficientViT_msra_224x224_nv12.bin`，而不是 Manifest 名称
   `EfficientViT_m5_224x224_nv12.bin`。要复现已发布制品，需把产出的
   `.bin` 改名为 Manifest 名称，或先修改前缀。（ONNX 输入名带 m5
   身份；只有输出侧无变体。）
4. **无固定的编译命令。**源 README 指向通用 OE 流程；产出已发布
   `.bin` 的确切命令没有记录，复现未经验证。
5. **本次迁移未执行任何转换。**
