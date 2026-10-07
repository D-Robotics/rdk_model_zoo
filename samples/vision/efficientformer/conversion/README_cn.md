# EfficientFormer 转换

本目录提供转换资产：两份参考 PTQ YAML
（`EfficientFormer_l1_config.yaml`、`EfficientFormer_l3_config.yaml`）。
编译前，按 YAML 指定路径准备模型 ONNX 图与校准数据，再使用下文 OE 命令。

<a id="source-model"></a>
## 源模型

EfficientFormer-L1 与 EfficientFormer-L3（论文 [EfficientFormer:
Vision Transformers at MobileNet
Speed](https://arxiv.org/abs/2206.01191)）。YAML 期望
`./efficientformer_l1.onnx` / `./efficientformer_l3.onnx`；导出时按模型变体准备匹配权重，并将图保存至对应 YAML 路径。

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机上的 RDK X5 OpenExplorer Docker 内执行
（march `bayes-e`），从不在板卡上运行。请准备含 `hb_mapper`、
`hb_perf`、`hrt_model_exec` 的工具链；离线
Docker 镜像可从 D-Robotics 开发者论坛
（[topic 35229](https://forum.d-robotics.cc/t/topic/35229)）获取。

<a id="export"></a>
## 导出

使用上游 EfficientFormer 流程和 `timm` 导出 ONNX：

1. 用 `timm.models.create_model` 创建目标 EfficientFormer 模型，如
   `efficientformer_l1` 或 `efficientformer_l3`。
2. 用 `torch.onnx.export` 导出模型。
3. 用 `onnxsim.simplify` 化简 ONNX 模型。
4. 在 OE 环境中编译化简后的 ONNX 模型（见"编译"）。

产物必须命名为 `efficientformer_l1.onnx` /
`efficientformer_l3.onnx` 并放在本目录（或修改 YAML 的 `onnx_model`）。

<a id="calibration"></a>
## 校准

按 YAML 准备校准数据： `./calibration_data_rgb_f32`
（float32 RGB `.npy`）。在 224x224 尺寸下按 YAML 数值处理数据：mean
`123.675 116.28 103.53`、scale `0.01712475 0.017507 0.01742919`。

<a id="compile"></a>
## 编译

在 OE 环境内、两个输入就绪后，编译对应变体：

```bash
# 输入：./efficientformer_l1.onnx + ./calibration_data_rgb_f32
# 输出前缀：EfficientFormer_224x224_nv12
hb_mapper checker --config EfficientFormer_l1_config.yaml
hb_mapper makertbin --config EfficientFormer_l1_config.yaml
```

两份 YAML 均使用 `calibration_type: 'default'` 加
`optimization: "set_all_nodes_int16"`、`compile_mode: 'latency'` /
`optimize_level: 'O3'`；L3 另设 `jobs: 64`，且比 L1 携带更多
`node_info` Softmax int16 配置。

<a id="validation"></a>
## 验证

使用 OE 包中的以下工具进行主机模型检查： `hb_perf` 与
`hrt_model_exec`。板端功能检查走样例运行时：
`python3 samples/vision/efficientformer/runtime/python/main.py --target x5 --asset-id x5:efficientformer:EfficientFormer_l1_224x224_nv12.bin...`
（见 [runtime/python/README_cn.md](../runtime/python/README_cn.md)）。
运行时期望的输入张量为 NV12 打包前的 `1x3x224x224`，输出为
ImageNet-1k 分类 logits。

<a id="artifacts"></a>
## 产物

使用 `EfficientFormer_l1_config.yaml` 或 `EfficientFormer_l3_config.yaml`
及其匹配的 ONNX 图和校准目录。

<a id="known-gaps"></a>
## 补充准备

从匹配的上游权重与导出流程准备 `efficientformer_l1.onnx` 或 `efficientformer_l3.onnx`。按所选 YAML 的 RGB/NCHW 归一化准备 `./calibration_data_rgb_f32`（mean `123.675 116.28 103.53`、scale `0.01712475 0.017507 0.01742919`）。两份 YAML 共用 `working_dir: 'EfficientFormer_224x224_nv12_int16'` 与 `output_model_file_prefix: 'EfficientFormer_224x224_nv12'`；每次在独立工作目录构建一个变体，并在部署前使用对应 Manifest 文件名。使用匹配 YAML 执行上文 OE 命令。
