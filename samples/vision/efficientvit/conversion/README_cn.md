# EfficientViT 转换

本目录提供转换资产：单份参考 PTQ YAML
（`EfficientViT_MSRA_m5_config.yaml`）。目录**不带导出脚本，也不带
校准数据生产脚本**，因此这是参考配置集而非完全可复现的流程；可复现
边界见[补充准备](#known-gaps)。

<a id="source-model"></a>
## 源模型

EfficientViT-MSRA m5（论文 [EfficientViT: Memory Efficient Vision
Transformer with Cascaded Group
Attention](https://arxiv.org/abs/2305.07027)，参考实现
[microsoft/Cream/EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT)）。
YAML 期望 `./efficientvit_m5.onnx`；导出时将匹配的 m5 模型图保存至此路径。

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机上的 RDK X5 OpenExplorer Docker（march
`bayes-e`）中执行，绝不在板卡上运行。请准备含 `hb_mapper`、`hb_perf`、
`hrt_model_exec` 的工具链；离线 Docker 镜像可从 D-Robotics 开发者论坛
（[topic 35229](https://forum.d-robotics.cc/t/topic/35229)）获取。

<a id="export"></a>
## 导出

目录不含导出脚本。原始 EfficientViT_MSRA 流程从 `timm` 实现导出
ONNX，导出参考如下：

1. 安装所需 Python 包，如 `timm`、`onnx`、`onnxsim`。
2. 创建预训练的 `efficientvit_m5` 模型。
3. 用 `1x3x224x224` 虚拟输入，通过 `torch.onnx.export` 导出模型。
4. 用 `onnxsim` 化简导出的 ONNX 模型。
5. 在 OE 环境中编译化简后的 ONNX 模型（见"编译"）。

产物必须命名为 `efficientvit_m5.onnx` 并放在本目录
（或修改 YAML 的 `onnx_model`）。

<a id="calibration"></a>
## 校准

准备 `./calibration_data_rgb_f32` 作为 224x224 输入的 float32 RGB `.npy`
数据。YAML 使用 `calibration_type: 'max'` 与
`max_percentile: 0.99999`，并按 YAML 归一化：mean `123.675 116.28
103.53`，scale `0.01712475 0.017507 0.01742919`。

<a id="compile"></a>
## 编译

在 OE 环境中，两个输入就绪后：

```bash
# cwd：本 conversion 目录
# 输入：./efficientvit_m5.onnx + ./calibration_data_rgb_f32
# 输出：working_dir 'EfficientViT_msra_224x224_nv12'，产出
#       EfficientViT_msra_224x224_nv12.bin；部署时使用 Manifest 文件名
hb_mapper checker --config EfficientViT_MSRA_m5_config.yaml
hb_mapper makertbin --config EfficientViT_MSRA_m5_config.yaml
```

YAML 设置 `compile_mode: 'latency'` / `optimize_level: 'O3'`，并通过
`node_info` 将 28 个注意力 `Softmax` 节点以 int16 I/O 摆上 BPU（级联组
注意力结构）。本配置不带 `debug_mode`，也没有 `set_all_nodes_int16`
优化。

<a id="validation"></a>
## 转换后验证

使用 OE 包中的以下工具进行主机模型检查： `hb_perf` 与
`hrt_model_exec`。板端功能检查走样例运行时：
`python3 samples/vision/efficientvit/runtime/python/main.py --target x5 --asset-id x5:efficientvit:EfficientViT_m5_224x224_nv12.bin...`
（见 [runtime/python/README_cn.md](../runtime/python/README_cn.md)）。
运行时期望的输入张量为 NV12 打包前的 `1x3x224x224`，输出为
ImageNet-1k 分类 logits。

<a id="artifacts"></a>
## 配方文件

参考 YAML `EfficientViT_MSRA_m5_config.yaml` 即本目录的转换资产；其
SHA-256 如下，可用于核对本地副本：

| 文件 | SHA-256 |
| --- | --- |
| `EfficientViT_MSRA_m5_config.yaml` | `65915fea82515c66457d1ac48051bd8172f97155e89dc9b9a65499dfcb13312c` |

<a id="known-gaps"></a>
## 补充准备

将 ONNX 图准备为 `./efficientvit_m5.onnx`，并在 `./calibration_data_rgb_f32` 准备 224x224 输入的 float32 RGB `.npy` 校准数据。使用 `calibration_type: 'max'`、`max_percentile: 0.99999`、mean `123.675 116.28 103.53` 和 scale `0.01712475 0.017507 0.01742919`。YAML 产出 `EfficientViT_msra_224x224_nv12.bin`；部署前将文件命名为与 Manifest 一致的 `EfficientViT_m5_224x224_nv12.bin`。使用上文命令运行 checker 与编译。
