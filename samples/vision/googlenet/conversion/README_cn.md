[English](README.md) | 简体中文

# GoogLeNet 转换

<a id="source-model"></a>
## 源模型

推理时按模型指南准备已发布的 X5 部署 bin
（`googlenet_224x224_nv12.bin`）。重新构建时先准备匹配的 ONNX 图、
权重修订与 PTQ YAML。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

在 x86 Linux 主机的 OpenExplorer Docker 或对应 OE 包编译环境中为
X5 构建模型。先准备 ONNX 图与 PTQ 配置，再使用 OE 包提供的
`hb_mapper checker`、`hb_mapper makertbin`、`hb_perf`、`hrt_model_exec`。
离线 Docker 镜像可从地瓜机器人开发者
论坛（[topic 35229](https://forum.d-robotics.cc/t/topic/35229)）获取。

<a id="export"></a>
## ONNX 导出

导出的 GoogLeNet ONNX 图需满足脚本的运行时输入输出要求：NV12 打包前的
RGB/NCHW 224×224 输入，以及 1000 类分数输出。

<a id="calibration"></a>
## 校准

针对所选模型图和量化配方准备校准图像及预处理数值，再运行 PTQ。

<a id="compile"></a>
## 编译

准备符合运行时输入输出要求的 ONNX 图与 PTQ 配置（见
[补充准备](#known-gaps)），再使用上文 OE 编译命令。推理请使用
[model/README_cn.md](../model/README_cn.md) 的发布制品路线。

<a id="validation"></a>
## 转换后验证

重建模型的运行时协议为：NV12 打包前 `1x3x224x224` 输入、
ImageNet-1k 分类 logits 输出（squeeze 后 1000 分数的单 F32 向量）。
重建模型可用对应模型引用加外部 `--model-path` 选择。发布制品的功能检查：

```bash
# cwd: repository root on X5
bash samples/vision/googlenet/model/download.sh x5 googlenet
python3 samples/vision/googlenet/runtime/python/main.py --target x5 --variant googlenet
```

<a id="artifacts"></a>
## 产物

发布制品及落地路径见[模型准备](../model/README_cn.md#artifacts)。

<a id="known-gaps"></a>
## 补充准备

推理时按 [model/download.sh](../model/README_cn.md) 准备已发布的 X5 模型。构建替代制品时，先准备符合[源模型](#source-model)运行时输入输出要求的 ONNX 图、匹配的校准数据与归一化配置，并为 X5 OE 工具链配置 PTQ YAML。编译后使用样例运行时运行该制品。

## 独立 TorchVision 0.21 crop224 规格

[上游声明](UPSTREAM_NOTICE.md) 固定 TorchVision commit、官方权重完整摘要和原始
BSD-3-Clause 文件。它使用新的独立文件与 variant，旧 X5 默认及其数据保留。

```bash
# 仓库根目录；Torch 2.6.0 CPU / TorchVision 0.21.0 / ORT 1.19.2。
python3 samples/vision/googlenet/conversion/export.py \
  --checkpoint /path/to/googlenet-1378be20.pth \
  --calibration /path/to/frozen/raw-rgb-calibration --output /path/to/new-run/export
```

先严格加载全部权重，保留 transform_input=True，再去掉不参与 eval 的辅助头。
三个真实图片和 seed-0 输入必须在 atol 1e-4 / rtol 1e-3 下通过原模型与 ORT 对照，
并保留 Top-5 排名。部署几何为 Pillow bicubic short256、rounded crop224 和
ImageNet mean/std；不引用上游其他预处理条件下的精度。S 校准采用归一化后的 ONNX
输入域张量；按[公共分类流程](../../../../utils/tools/classification_conversion/README.md)
逐芯片生成 PTQ、BC、完整 NodeInfo 和最终 HBM/OE。板端数据保持 `not-run`。

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

每个目标使用自己的 PTQ、BC、完整 NodeInfo 和最终 OE。内部 INT16 配置不能替代实际公共 dtype；本轮按已读取的 float32 输出绑定。具体配方及 SHA 见 [converted_targets_20261103.json](converted_targets_20261103.json)。

- `s100 / torchvision-021-crop224-int16-o1-20261103 / model`: [torchvision_021_crop224_int16_o1_20261103_s100_model.yaml](recipes/torchvision_021_crop224_int16_o1_20261103_s100_model.yaml)
- `s100p / torchvision-021-crop224-int16-o1-20261103 / model`: [torchvision_021_crop224_int16_o1_20261103_s100p_model.yaml](recipes/torchvision_021_crop224_int16_o1_20261103_s100p_model.yaml)
- `s600 / torchvision-021-crop224-int16-o1-20261103 / model`: [torchvision_021_crop224_int16_o1_20261103_s600_model.yaml](recipes/torchvision_021_crop224_int16_o1_20261103_s600_model.yaml)

S 校准必须使用 ONNX 输入域的归一化 `.npy`；NV12 Runtime 的 mean/scale 仍写在配方中。公共工具的 `--target` 和 `--normalization-json` 从同一 checkpoint 导出记录读取实际归一化。

[Shared classification workflow](../../../../utils/tools/classification_conversion/README.md).

原始浮点门禁的阈值、真实输入和 checkpoint 绑定保持不变。编译估算及首条校准输入的 NodeInfo 诊断不作为数据集精度或板端性能。
